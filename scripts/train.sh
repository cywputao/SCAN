python -W ignore -u main.py \
--train \
--test \
--data "$2" \
--enc_dim "$3" \
--dec_dim "$4" \
--att_dim "$5" \
--model "$1" \
--batch_size 64 \
--embedding_dim "$6" \
--epochs 200 \
--scheduler \
--delim tab \
--learning_rate 1e-03 \

