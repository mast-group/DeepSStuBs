import json
import sys

print(sys.argv[1])
with open(sys.argv[1], "r") as read_file:
    sstubs = json.load(read_file)

print('Loaded %d sstubs' % len(sstubs))

change_operators = 0
change_operands = 0
swap_args = 0

with open('swap_args_filter', 'w') as f_arg, open('change_ops_filter', 'w') as f_op, open('change_operands_filter', 'w') as f_operand:
    for sstub in sstubs:
        if sstub['bugType'] == 'CHANGE_OPERATOR':
            change_operators += 1
            f_op.write('%d %d %s %s' % (change_operators, sstub['lineNum'], sstub['before'].replace(' ', ''), sstub['after'].replace(' ', '')))
            f_op.write('\n')
        elif sstub['bugType'] == 'CHANGE_OPERAND':
            change_operands += 1
            s = '%d %d %s %s' % (change_operands, sstub['lineNum'], sstub['before'].replace(' ', ''), sstub['after'].replace(' ', ''))
            f_operand.write(s.encode("utf-8"))
            f_operand.write('\n')
        elif sstub['bugType'] == 'SWAP_ARGUMENTS':
            swap_args += 1
            f_arg.write('%d %d %s %s' % (swap_args, sstub['lineNum'], sstub['before'].replace(' ', ''), sstub['after'].replace(' ', '')))
            f_arg.write('\n')
