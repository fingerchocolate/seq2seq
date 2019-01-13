#python3 seq2seq3_test.py --gpu=0 original_test_prepro.txt simple_test_prepro.txt \vocab_train.txt vocab_train.txt \-r (学習済みmodelのsnapshot from seq2seq3.py) \-output (生成文とbleuを書き込むtxt)
#train用のseq2seq3.pyからの学習済みモデルを読み込み, testデータから生成文&bleuスコア出す
from __future__ import division #sari
import argparse
import datetime

from nltk.translate import bleu_score
#import nltk
import numpy
import progressbar
import six

import chainer
from chainer.backends import cuda
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

import os


from collections import Counter #sari
import sys #sari


UNK = 0
EOS = 1


def sequence_embed(embed, xs):
    x_len = [len(x) for x in xs]
    x_section = numpy.cumsum(x_len[:-1])
    ex = embed(F.concat(xs, axis=0))
    exs = F.split_axis(ex, x_section, 0)
    return exs


class Seq2seq(chainer.Chain):

    def __init__(self, n_layers, n_source_vocab, n_target_vocab, n_units):
        super(Seq2seq, self).__init__()
        with self.init_scope():
            self.embed_x = L.EmbedID(n_source_vocab, n_units)
            self.embed_y = L.EmbedID(n_target_vocab, n_units)
            self.encoder = L.NStepLSTM(n_layers, n_units, n_units, 0.1)
            self.decoder = L.NStepLSTM(n_layers, n_units, n_units, 0.1)
            self.W = L.Linear(n_units, n_target_vocab)

        self.n_layers = n_layers
        self.n_units = n_units

    def forward(self, xs, ys):
        xs = [x[::-1] for x in xs]

        eos = self.xp.array([EOS], numpy.int32)
        ys_in = [F.concat([eos, y], axis=0) for y in ys]
        ys_out = [F.concat([y, eos], axis=0) for y in ys]

        # Both xs and ys_in are lists of arrays.
        exs = sequence_embed(self.embed_x, xs)
        eys = sequence_embed(self.embed_y, ys_in)

        batch = len(xs)
        # None represents a zero vector in an encoder.
        hx, cx, _ = self.encoder(None, None, exs)
        _, _, os = self.decoder(hx, cx, eys)

        # It is faster to concatenate data before calculating loss
        # because only one matrix multiplication is called.
        concat_os = F.concat(os, axis=0)
        concat_ys_out = F.concat(ys_out, axis=0)
        loss = F.sum(F.softmax_cross_entropy(
            self.W(concat_os), concat_ys_out, reduce='no')) / batch

        chainer.report({'loss': loss}, self)
        n_words = concat_ys_out.shape[0]
        perp = self.xp.exp(loss.array * batch / n_words)
        chainer.report({'perp': perp}, self)
        return loss

    def translate(self, xs, max_length=100):
        batch = len(xs)
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            xs = [x[::-1] for x in xs]
            exs = sequence_embed(self.embed_x, xs)
            h, c, _ = self.encoder(None, None, exs)
            ys = self.xp.full(batch, EOS, numpy.int32)
            result = []
            for i in range(max_length):
                eys = self.embed_y(ys)
                eys = F.split_axis(eys, batch, 0)
                h, c, ys = self.decoder(h, c, eys)
                cys = F.concat(ys, axis=0)
                wy = self.W(cys)
                ys = self.xp.argmax(wy.array, axis=1).astype(numpy.int32)
                result.append(ys)

        # Using `xp.concatenate(...)` instead of `xp.stack(result)` here to
        # support NumPy 1.9.
        result = cuda.to_cpu(
            self.xp.concatenate([self.xp.expand_dims(x, 0) for x in result]).T)

        # Remove EOS taggs
        outs = []
        for y in result:
            inds = numpy.argwhere(y == EOS)
            if len(inds) > 0:
                y = y[:inds[0, 0]]
            outs.append(y)
        return outs


def convert(batch, device):
    def to_device_batch(batch):
        if device is None:
            return batch
        elif device < 0:
            return [chainer.dataset.to_device(device, x) for x in batch]
        else:
            xp = cuda.cupy.get_array_module(*batch)
            concat = xp.concatenate(batch, axis=0)
            sections = numpy.cumsum([len(x)
                                     for x in batch[:-1]], dtype=numpy.int32)
            concat_dev = chainer.dataset.to_device(device, concat)
            batch_dev = cuda.cupy.split(concat_dev, sections)
            return batch_dev

    return {'xs': to_device_batch([x for x, _ in batch]),
            'ys': to_device_batch([y for _, y in batch])}


def count_lines(path):
    with open(path) as f:
        return sum([1 for _ in f])


def load_vocabulary(path):
    with open(path) as f:
        # +2 for UNK and EOS
        word_ids = {line.strip(): i + 2 for i, line in enumerate(f)}
    word_ids['<UNK>'] = 0
    word_ids['<EOS>'] = 1
    return word_ids

def load_data_using_dataset_api(
        src_vocab, src_path, target_vocab, target_path, filter_func):

    def _transform_line(vocabulary, line):
        words = line.strip().split()
        return numpy.array(
            [vocabulary.get(w, UNK) for w in words], numpy.int32)

    def _transform(example):
        source, target = example
        return (
            _transform_line(src_vocab, source),
            _transform_line(target_vocab, target)
        )

    return chainer.datasets.TransformDataset(
        chainer.datasets.TextDataset(
            [src_path, target_path],
            encoding='utf-8',
            filter_func=filter_func
        ), _transform)


def calculate_unknown_ratio(data):
    unknown = sum((s == UNK).sum() for s in data)
    total = sum(s.size for s in data)
    return unknown / total

def main():
    parser = argparse.ArgumentParser(description='Chainer example: seq2seq')
    parser.add_argument('SOURCE', help='source sentence list')
    parser.add_argument('TARGET', help='target sentence list')
    parser.add_argument('SOURCE_VOCAB', help='source vocabulary file')
    parser.add_argument('TARGET_VOCAB', help='target vocabulary file')
#    parser.add_argument('--batchsize', '-b', type=int, default=64,
    parser.add_argument('--batchsize', '-b', type=int, default=32,##
                        help='number of sentence pairs in each mini-batch')
#    parser.add_argument('--epoch', '-e', type=int, default=20,
    parser.add_argument('--epoch', '-e', type=int, default=40,##
                        help='number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--resume', '-r', default='',
                        help='resume the training from snapshot')
#    parser.add_argument('--unit', '-u', type=int, default=1024,
    parser.add_argument('--unit', '-u', type=int, default=300,
                        help='number of units')
#    parser.add_argument('--layer', '-l', type=int, default=3,
    parser.add_argument('--layer', '-l', type=int, default=2,##
                        help='number of layers')
#    parser.add_argument('--use-dataset-api', default=False,
    parser.add_argument('--use-dataset-api', default=True,
                        action='store_true',
                        help='use TextDataset API to reduce CPU memory usage')
    parser.add_argument('--min-source-sentence', type=int, default=1,
                        help='minimium length of source sentence')
#    parser.add_argument('--max-source-sentence', type=int, default=50,
    parser.add_argument('--max-source-sentence', type=int, default=14,##
                        help='maximum length of source sentence')
    parser.add_argument('--min-target-sentence', type=int, default=1,
                        help='minimium length of target sentence')
#    parser.add_argument('--max-target-sentence', type=int, default=50,
    parser.add_argument('--max-target-sentence', type=int, default=14,##
                        help='maximum length of target sentence')
    parser.add_argument('--log-interval', type=int, default=200,
                        help='number of iteration to show log')
#    parser.add_argument('--validation-interval', type=int, default=4000,
    parser.add_argument('--validation-interval', type=int, default=200,
                        help='number of iteration to evlauate the model '
                        'with validation dataset')
    parser.add_argument('--out', '-o', default='result',
                        help='directory to output the result')
    parser.add_argument('--output', default='text_and_bleu.txt',
                        help='test and bleu output file')
    args = parser.parse_args()

    # Load pre-processed dataset
    print('[{}] Loading dataset... (this may take several minutes)'.format(
        datetime.datetime.now()))
    source_ids = load_vocabulary(args.SOURCE_VOCAB)
    target_ids = load_vocabulary(args.TARGET_VOCAB)

    # By using TextDataset, you can avoid loading whole dataset on memory. ##
    # This significantly reduces the host memory usage. ##
    def _filter_func(s, t):##
        sl = len(s.strip().split())  # number of words in source line ##
        tl = len(t.strip().split())  # number of words in target line ##
        return (
            0 < sl and ##
            0 < tl ) ##

    test_data = load_data_using_dataset_api( ##
        source_ids, args.SOURCE, ##
        target_ids, args.TARGET, ##
        _filter_func,##
    ) ##

    target_words = {i: w for w, i in target_ids.items()}
    source_words = {i: w for w, i in source_ids.items()}

    # Setup model
    model = Seq2seq(args.layer, len(source_ids), len(target_ids), args.unit)


#    os.path.listdir(args.resume)
#    if args.resume:
#        # Resume from a snapshot
    chainer.serializers.load_npz(args.resume, model)

    if args.gpu >= 0:
        chainer.backends.cuda.get_device(args.gpu).use()
        model.to_gpu(args.gpu)

#    # Setup optimizer
#    optimizer = chainer.optimizers.Adam()
#    optimizer.setup(model)

#    # Setup iterator
#    test_iter = chainer.iterators.SerialIterator(test_data, args.batchsize)

#    # Setup updater and trainer
#    updater = training.updaters.StandardUpdater(
#        test_iter, optimizer, converter=convert, device=args.gpu)
#    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    path_w = args.output
    with open(path_w, 'w'):
        pass


    def SARIngram(sgrams, cgrams, rgramslist, numref):

        rgramsall = [rgram for rgrams in rgramslist for rgram in rgrams]
        rgramcounter = Counter(rgramsall)

        sgramcounter = Counter(sgrams)
        sgramcounter_rep = Counter()
        for sgram, scount in sgramcounter.items():
            sgramcounter_rep[sgram] = scount * numref

        cgramcounter = Counter(cgrams)
        cgramcounter_rep = Counter()
        for cgram, ccount in cgramcounter.items():
            cgramcounter_rep[cgram] = ccount * numref


        # KEEP
        keepgramcounter_rep = sgramcounter_rep & cgramcounter_rep
        keepgramcountergood_rep = keepgramcounter_rep & rgramcounter
        keepgramcounterall_rep = sgramcounter_rep & rgramcounter

        keeptmpscore1 = 0
        keeptmpscore2 = 0
        for keepgram in keepgramcountergood_rep:
            keeptmpscore1 += keepgramcountergood_rep[keepgram] / keepgramcounter_rep[keepgram]
            keeptmpscore2 += keepgramcountergood_rep[keepgram] / keepgramcounterall_rep[keepgram]
            #print "KEEP", keepgram, keepscore, cgramcounter[keepgram], sgramcounter[keepgram], rgramcounter[keepgram]
        keepscore_precision = 0
        if len(keepgramcounter_rep) > 0:
    	       keepscore_precision = keeptmpscore1 / len(keepgramcounter_rep)
        keepscore_recall = 0
        if len(keepgramcounterall_rep) > 0:
    	       keepscore_recall = keeptmpscore2 / len(keepgramcounterall_rep)
        keepscore = 0
        if keepscore_precision > 0 or keepscore_recall > 0:
            keepscore = 2 * keepscore_precision * keepscore_recall / (keepscore_precision + keepscore_recall)


        # DELETION
        delgramcounter_rep = sgramcounter_rep - cgramcounter_rep
        delgramcountergood_rep = delgramcounter_rep - rgramcounter
        delgramcounterall_rep = sgramcounter_rep - rgramcounter

        deltmpscore1 = 0
        deltmpscore2 = 0
        for delgram in delgramcountergood_rep:
            deltmpscore1 += delgramcountergood_rep[delgram] / delgramcounter_rep[delgram]
            deltmpscore2 += delgramcountergood_rep[delgram] / delgramcounterall_rep[delgram]
        delscore_precision = 0
        if len(delgramcounter_rep) > 0:
    	       delscore_precision = deltmpscore1 / len(delgramcounter_rep)
        delscore_recall = 0
        if len(delgramcounterall_rep) > 0:
    	       delscore_recall = deltmpscore1 / len(delgramcounterall_rep)
        delscore = 0
        if delscore_precision > 0 or delscore_recall > 0:
            delscore = 2 * delscore_precision * delscore_recall / (delscore_precision + delscore_recall)


        # ADDITION
        addgramcounter = set(cgramcounter) - set(sgramcounter)
        addgramcountergood = set(addgramcounter) & set(rgramcounter)
        addgramcounterall = set(rgramcounter) - set(sgramcounter)

        addtmpscore = 0
        for addgram in addgramcountergood:
            addtmpscore += 1

        addscore_precision = 0
        addscore_recall = 0
        if len(addgramcounter) > 0:
    	       addscore_precision = addtmpscore / len(addgramcounter)
        if len(addgramcounterall) > 0:
    	    addscore_recall = addtmpscore / len(addgramcounterall)
        addscore = 0
        if addscore_precision > 0 or addscore_recall > 0:
            addscore = 2 * addscore_precision * addscore_recall / (addscore_precision + addscore_recall)

        return (keepscore, delscore_precision, addscore)


#    def SARIsent (ssent, csent, rsents) :
    def SARIsent (s1grams, c1grams, r1grams) :#
#        numref = len(rsents)
        numref = 1 #
#        s1grams = ssent.lower().split(" ")
        ###['about', '95', 'species', 'are', 'currently', 'accepted', '.']
#        c1grams = csent.lower().split(" ")
        s2grams = []
        c2grams = []
        s3grams = []
        c3grams = []
        s4grams = []
        c4grams = []

        r1gramslist = []
        r2gramslist = []
        r3gramslist = []
        r4gramslist = []
#        for rsent in rsents:
#            r1grams = rsent.lower().split(" ")
        r2grams = []
        r3grams = []
        r4grams = []
        r1gramslist.append(r1grams)
        for i in range(0, len(r1grams)-1) :
            if i < len(r1grams) - 1:
                r2gram = r1grams[i] + " " + r1grams[i+1]
                r2grams.append(r2gram)
            if i < len(r1grams)-2:
                r3gram = r1grams[i] + " " + r1grams[i+1] + " " + r1grams[i+2]
                r3grams.append(r3gram)
            if i < len(r1grams)-3:
                r4gram = r1grams[i] + " " + r1grams[i+1] + " " + r1grams[i+2] + " " + r1grams[i+3]
                r4grams.append(r4gram)
        r2gramslist.append(r2grams)
        r3gramslist.append(r3grams)
        r4gramslist.append(r4grams)

        ###r2gramslist=[['about 95', '95 species', 'species are', 'are currently', 'currently known', 'known .'], ['about 95', '95 species', 'species are', 'are now', 'now accepted', 'accepted .'], ['95 species', 'species are', 'are now', 'now accepted', 'accepted .']]

        for i in range(0, len(s1grams)-1) :
            if i < len(s1grams) - 1:
                s2gram = s1grams[i] + " " + s1grams[i+1]
                s2grams.append(s2gram)
            if i < len(s1grams)-2:
                s3gram = s1grams[i] + " " + s1grams[i+1] + " " + s1grams[i+2]
                s3grams.append(s3gram)
            if i < len(s1grams)-3:
                s4gram = s1grams[i] + " " + s1grams[i+1] + " " + s1grams[i+2] + " " + s1grams[i+3]
                s4grams.append(s4gram)

        for i in range(0, len(c1grams)-1) :
            if i < len(c1grams) - 1:
                c2gram = c1grams[i] + " " + c1grams[i+1]
                c2grams.append(c2gram)
            if i < len(c1grams)-2:
                c3gram = c1grams[i] + " " + c1grams[i+1] + " " + c1grams[i+2]
                c3grams.append(c3gram)
            if i < len(c1grams)-3:
                c4gram = c1grams[i] + " " + c1grams[i+1] + " " + c1grams[i+2] + " " + c1grams[i+3]
                c4grams.append(c4gram)


        (keep1score, del1score, add1score) = SARIngram(s1grams, c1grams, r1gramslist, numref)
        (keep2score, del2score, add2score) = SARIngram(s2grams, c2grams, r2gramslist, numref)
        (keep3score, del3score, add3score) = SARIngram(s3grams, c3grams, r3gramslist, numref)
        (keep4score, del4score, add4score) = SARIngram(s4grams, c4grams, r4gramslist, numref)

        avgkeepscore = sum([keep1score,keep2score,keep3score,keep4score])/4
        avgdelscore = sum([del1score,del2score,del3score,del4score])/4
        avgaddscore = sum([add1score,add2score,add3score,add4score])/4
        finalscore = ( avgkeepscore + avgdelscore + avgaddscore ) / 3
        print(finalscore)#
        with open(path_w, mode='a') as f:#
            f.write('\n sari : %f' %finalscore)#

        #return finalscore

#    @chainer.training.make_extension()
    def translate(test_data):
#        source, target = test_data[numpy.random.choice(len(test_data))]
        for i in range(len(test_data)):
            source, target = test_data[i]
            result = model.translate([model.xp.array(source)])[0]
            #source_sentence = ' '.join([source_words[x] for x in source])
            #target_sentence = ' '.join([target_words[y] for y in target])
            #result_sentence = ' '.join([target_words[y] for y in result])
            list_source_sentence = [source_words[x] for x in source]#
            list_target_sentence = [target_words[y] for y in target]#
            list_result_sentence = [target_words[y] for y in result]#
            source_sentence = ' '.join(list_source_sentence)#
            target_sentence = ' '.join(list_target_sentence)#
            result_sentence = ' '.join(list_result_sentence)#
#            print('# source : ' + source_sentence)###
#            print('# result : ' + result_sentence)###
#            print('# expect : ' + target_sentence)###

            with open(path_w, mode='a') as f:
                f.write('\n source')
                f.write(source_sentence)
                f.write('\n target')
                f.write(target_sentence)
                f.write('\n result')
                f.write(result_sentence)

            SARIsent(list_source_sentence, list_target_sentence, list_result_sentence)#

#    @chainer.training.make_extension()
    def CalculateBleu(model=model, test_data=test_data, batch = 100, max_length=100, device=args.gpu):
        trigger = 1, 'epoch'
        priority = chainer.training.PRIORITY_WRITER
        with chainer.no_backprop_mode():
            references = []
            hypotheses = []
            for i in range(0, len(test_data), batch):
                sources, targets = zip(*test_data[i:i + batch])
                references.extend([[t.tolist()] for t in targets])

                sources = [
                    chainer.dataset.to_device(device, x) for x in sources]
                ys = [y.tolist()
                      for y in model.translate(sources, max_length)]
                hypotheses.extend(ys)
            print(len(references))
            bleu = bleu_score.corpus_bleu(
                references, hypotheses,
                smoothing_function=bleu_score.SmoothingFunction().method1)
            print(bleu)#
            with open(path_w, mode='a') as f:#
                f.write('\n bleu : %f' %bleu)#




    translate(test_data)
    CalculateBleu(model=model, test_data=test_data, batch = 100, max_length=100, device=args.gpu)

#    trainer.extend(
#        translate, trigger=(args.validation_interval, 'iteration'))

#    trainer.extend(
#        CalculateBleu, trigger=(args.validation_interval, 'iteration'))



if __name__ == '__main__':
    main()
