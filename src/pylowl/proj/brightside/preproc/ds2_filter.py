#!/usr/bin/env python


from pylowl.proj.brightside.utils import write_concrete


def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_path', type=str,
                        help='doc-per-line input file path')
    parser.add_argument('output_path', type=str,
                        help='doc-per-line output directory path')

    args = parser.parse_args()
    ds2_filter(
        args.input_path,
        args.output_path,
    )


def iter_docs(input_path):
    with open(input_path) as f:
        for line in f:
            pieces = line.strip().split('\t')
            user = pieces[0]
            datetime = pieces[1]
            text = '\t'.join(pieces[5:])
            yield dict(
                text=text,
                user=user,
                datetime=datetime,
            )


def format_to_concrete(input_path, output_path):
    write_concrete(iter_docs(input_path), output_path)


if __name__ == '__main__':
    main()
