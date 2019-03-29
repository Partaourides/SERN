import argparse
from emotion_recognition import emotion_recognition
from emotion_recognition_with_dialog_dependence import model_with_dialog_dependence

parser = argparse.ArgumentParser()

parser.add_argument('--model',
                    help='Model type: RNN, RNN_with_memory, Double_RNN_with_memory',
                    nargs='?',
                    default='Double_RNN_with_memory')
parser.add_argument('--n_run', help='Number of run iteration', nargs='?', type=int, default=50)
parser.add_argument('--score_func', help='Score function: dot, general, concat', nargs='?', default='dot')
parser.add_argument('--embedding', help='Embedding type: random glove', nargs='?', default='random')
parser.add_argument('--embedding_size', help='Embedding Size: If embedding=glove : 50, 100, 200, 300', nargs='?', type=int, default=50)
parser.add_argument('--first_rnn_size', help='The size of the first RNN', nargs='?', type=int, default=50)
parser.add_argument('--second_rnn_size', help='The size of the second RNN', nargs='?', type=int, default=50)
parser.add_argument('--window_size', help='The size of the window with -1 the absence of one', nargs='?', default='-1')
parser.add_argument('--epochs', help='Number of epochs', nargs='?', type=int, default=50)
parser.add_argument('--batch_size', help='Batch Size', nargs='?', type=int, default=20)
parser.add_argument('--dropout', help='Dropout', nargs='?', type=float, default=0.5)
parser.add_argument('--num_speakers', help='The ID of speakers as auxiliary', nargs='?', type=bool, default=False)
parser.add_argument('--primal_emotion', help='emotion that will be against all (off, ang, fru, sad)', nargs='?', default='off')
parser.add_argument('--num_desired_classes', nargs='?', type=int, default=6)

args = parser.parse_args()

if args.model == 'RNN':
    emotion_recognition(args.n_run,
                        args.epochs,
                        65 * args.batch_size,
                        args.embedding_size,
			            args.first_rnn_size,
                        args.dropout,
                        args.embedding,
                        args.num_speakers)

if args.model == 'Double_RNN' or args.model == 'Double_RNN_with_memory' or args.model == 'RNN_with_memory':
    model_with_dialog_dependence(args.n_run,
                                 args.epochs,
                                 args.batch_size,
                                 args.embedding_size,
                                 args.first_rnn_size,
                                 args.second_rnn_size,
                                 args.dropout,
                                 args.score_func,
                                 args.model,
                                 args.window_size,
                                 args.embedding,
                                 args.num_speakers,
                                 args.primal_emotion,
                                 args.num_desired_classes)
