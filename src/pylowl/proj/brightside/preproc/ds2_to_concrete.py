#!/usr/bin/env python


from pylowl.proj.brightside.corpus import write_concrete, Document


def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_path', type=str,
                        help='doc-per-line input file path')
    parser.add_argument('output_path', type=str,
                        help='doc-per-line output directory path')

    args = parser.parse_args()
    ds2_to_concrete(
        args.input_path,
        args.output_path,
    )


def iter_docs(input_path):
    with open(input_path) as f:
        for (line_num, line) in enumerate(f):
            pieces = line.strip().split('\t')
            user = pieces[0]
            datetime = pieces[1]
            latitude = pieces[3]
            longitude = pieces[3]
            text = '\t'.join(pieces[5:])
            tokens = [t for t in text.split()
                      if not [c for c in t if ord(c) > 127]]
            text = ' '.join(tokens)
            yield Document(tokens,
                text=text,
                user=user,
                datetime=datetime,
                latitude=latitude,
                longitude=longitude,
                id=str(line_num),
            )


def ds2_to_concrete(input_path, output_path):
    write_concrete(iter_docs(input_path), output_path)


if __name__ == '__main__':
    main()
