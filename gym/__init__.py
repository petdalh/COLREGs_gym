# This package shadows the installed `gym` (OpenAI/Gymnasium) package when
# the project root is on sys.path. SB3 checks for gym.__version__ at save
# time — expose a stub so that check doesn't raise AttributeError.
__version__ = "0.0.0"
