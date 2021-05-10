from collections import Counter
import csv
import subprocess
from util import inflect
import pandas as pd
from statsmodels.stats.proportion import proportion_confint
infl_eng = inflect.engine()

dependency_fields = ['sentence', 'orig_sentence', 'pos_sentence',
                     'subj', 'verb', 'subj_pos', 'has_rel', 'has_nsubj',
                     'verb_pos', 'subj_index', 'verb_index', 'n_intervening',
                     'last_intervening', 'n_diff_intervening', 'distance',
                     'max_depth', 'all_nouns', 'nouns_up_to_verb']


def deps_to_tsv(deps, outfile):
    writer = csv.writer(open(outfile, 'w'), delimiter='\t')
    writer.writerow(dependency_fields)
    for dep in deps:
        writer.writerow([dep[key] for key in dependency_fields])


def deps_from_tsv(infile, limit=None):
    res = []
    for i, d in enumerate(csv.DictReader(open(infile), delimiter='\t')):
        if limit is not None and i >= limit:
            break
        res.append({x: int(y) if y.isdigit() else y for x, y in d.items()})
    return res


def zread(fname):
    p = subprocess.Popen(['gunzip', '-c', fname], stdout=subprocess.PIPE)
    for line in p.stdout:
        yield line
    p.wait()


def tokenize_blanks(fh):
    sent = []
    for line in fh:
        line = line.strip().split()
        if not line:
            if sent:
                yield sent
            sent = []
        else:
            sent.append(line)
    yield sent


def create_freq_dict(infile, outfile, minfreq=50):
    d = Counter()
    for i, line in enumerate(zread(infile)):
        stripped = line.strip()
        if stripped:
            s = stripped.split()
            d[s[1], s[3]] += 1
        if i % 1000000 == 0:
            print(i)

    outfile = file(outfile, 'w')
    for (w, pos), count in d.iteritems():
        if count > minfreq:
            outfile.write('%s\t%s\t%d\n' % (w, pos, count))


def confint(row):
    n_errors = int(row['errorprob'] * row['count'])
    return proportion_confint(n_errors, row['count'])


def add_confints(df):
    df['minconf'] = df.apply(lambda row: confint(row)[0], axis=1)
    df['maxconf'] = df.apply(lambda row: confint(row)[1], axis=1)


def get_grouping(df, grouping_vars):
    funcs = {'correct': {'accuracy': 'mean', 'count': 'count'},
             'distance': {'mean_distance': 'mean'}}
    x = df.groupby(grouping_vars).aggregate(funcs)
    x.columns = x.columns.droplevel()
    x = x.reset_index()
    x['errorprob'] = 1 - x['accuracy']
    add_confints(x)
    return x

def gen_inflect_from_vocab(vocab_file, freq_threshold=1000):
    vbp = {}
    vbz = {}
    nn = {}
    nns = {}
    from_pos = {'NNS': nns, 'NN': nn, 'VBP': vbp, 'VBZ': vbz}

    for line in open(vocab_file):
        if line.startswith(' '):   # empty string token
            continue
        word, pos, count = line.strip().split()
        count = int(count)
        if len(word) > 1 and pos in from_pos and count >= freq_threshold:
            from_pos[pos][word] = count

    verb_infl = {'VBP': 'VBZ', 'VBZ': 'VBP'}
    for word, count in vbz.items():
        candidate = infl_eng.plural_verb(word)
        if candidate in vbp:
            verb_infl[candidate] = word
            verb_infl[word] = candidate

    noun_infl = {'NN': 'NNS', 'NNS': 'NN'}
    for word, count in nn.items():
        candidate = infl_eng.plural_noun(word)
        if candidate in nns:
            noun_infl[candidate] = word
            noun_infl[word] = candidate

    return verb_infl, noun_infl

def annotate_relpron(df):
    pd.options.mode.chained_assignment = None

    def f(x):
        blacklist = set(['NNP', 'PRP'])
        relprons = set(['WDT', 'WP', 'WRB', 'WP$'])
        vi = x['verb_index'] - 1
        words_in_dep = x['orig_sentence'].split()[x['subj_index']:vi]
        pos_in_dep = x['pos_sentence'].split()[x['subj_index']:vi]
        first_is_that = words_in_dep[:1] == ['that']
        return (bool(blacklist & set(pos_in_dep)),
                bool(relprons & set(pos_in_dep[:2])) | first_is_that,
                bool(relprons & set(pos_in_dep)) | first_is_that)

    df['blacklisted'], df['has_early_relpron'], df['has_relpron'] = \
        zip(*df.apply(f, axis=1))
    df['has_early_relpron'] = True

    def g(x):
        if x['has_rel'] and x['has_relpron'] and x['has_early_relpron']:
            return 'With relativizer'
        elif x['has_rel'] and not x['has_relpron']:
            return 'Without relativizer'
        elif not x['has_rel']:
            if x['has_relpron']:
                return 'Error'
            else:
                return 'No relative clause'
        else:
            return 'Error'

    df['condition'] = df.apply(g, axis=1)
    return df