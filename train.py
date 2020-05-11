# from encoder import get_encoder
from encoder import get_encoder

if __name__ == '__main__':
    encoder = get_encoder()
    print(encoder.encode('class Model(object):'))
