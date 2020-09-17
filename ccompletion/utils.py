from pathlib import Path

import chardet


def read_source_file(path: Path) -> str:
    try:
        # attempt to read file using utf-8 encoding
        with open(path, 'r', encoding='utf-8', errors='strict') as fd:
            return fd.read().strip(), 'utf-8'
    except UnicodeDecodeError:
        # determine encoding of file using chardet module
        with open(path, 'rb') as fd:
            result = chardet.detect(fd.read())
            encoding = result['encoding']

            # rethrow error to caller if encoding could not be determined
            if encoding is None:
                raise

        # open file using the guessed encoding
        # errors raised on this one will be rethrown back to the caller
        with open(path, 'r', encoding=encoding, errors='strict') as fd:
            return fd.read().strip(), encoding
