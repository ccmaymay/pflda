import os
import codecs


def load_word_set(filename):
    word_set = set()
    with codecs.open(filename, encoding='utf-8') as f:
        for line in f:
            word_set.add(line.strip())
    return word_set


def write_vocab(output_filename, vocab):
    make_parent_dir(output_filename)
    with codecs.open(output_filename, mode='w', encoding='utf-8') as out_f:
        for (word, word_id) in vocab.items():
            out_f.write(u'%d %s\n' % (word_id, word))


def get_path_suffix(path, stem):
    path_stem = os.path.normpath(path)
    path_suffix = None
    while os.path.normpath(os.path.abspath(path_stem)) != os.path.normpath(os.path.abspath(stem)):
        if not path_stem:
            raise Exception('"%s" is not an ancestor of "%s"'
                            % (stem, path))

        (path_stem, basename) = os.path.split(path_stem)
        if path_suffix is None:
            path_suffix = basename
        else:
            path_suffix = os.path.join(basename, path_suffix)
    if path_suffix is None:
        return os.path.curdir
    else:
        return path_suffix


def make_parent_dir(path):
    parent_path = os.path.dirname(path)
    if not os.path.isdir(parent_path):
        os.makedirs(parent_path)


def input_output_paths(input_path, output_path):
    if os.path.isdir(input_path):
        for (dir_path, dir_entries, file_entries) in os.walk(input_path):
            rel_dir_path = get_path_suffix(dir_path, input_path)
            for filename in file_entries:
                input_file_path = os.path.join(dir_path, filename)
                output_dir_path = os.path.join(output_path, rel_dir_path)
                output_file_path = os.path.join(output_dir_path, filename)
                make_parent_dir(output_file_path)
                yield (input_file_path, output_file_path)
    elif os.path.isfile(input_path):
        make_parent_dir(output_path)
        yield (input_path, output_path)
    else:
        raise Exception('"%s" does not seem to be a valid file or directory'
                        % input_path)
