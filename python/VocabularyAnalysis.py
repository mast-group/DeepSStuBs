
import json
import os
import re
import sys

from gensim.models import Word2Vec 

calls_with_two_args = 0
no_base = 0
base_OOV = 0
callee_OOV = 0
first_arg_OOV = 0
second_arg_OOV = 0
both_args_OOV = 0
base_and_callee_OOV = 0
base_and_args_OOV = 0
callee_and_one_arg_OOV = 0
callee_and_args_OOV = 0
all_call_parts_OOV = 0


bin_ops = 0
left_operand_OOV = 0
right_operand_OOV = 0
both_operands_OOV = 0
left_op_type_unk = 0
right_op_type_unk = 0
both_op_types_unk = 0
all_op_parts_unk = 0

voc = set()


def iterate_instances(directory, what, filetype):
    PATTERN = re.compile('%s_%s_[0-9]+\.json' % (what, filetype))

    files = os.listdir(directory)
    for file in files:
        if PATTERN.fullmatch(file):
            yield '%s/%s' % (directory, file)


def analyze_binOp(file, model):
    global bin_ops, left_operand_OOV, right_operand_OOV, both_operands_OOV
    global left_op_type_unk, right_op_type_unk, both_op_types_unk, all_op_parts_unk
    
    with open(file, 'r') as f:
        json_data = f.read()
        bin_ops_data = json.loads(json_data)
        for bin_op in bin_ops_data:
            bin_ops += 1
            left_op_type = bin_op['leftType']
            right_op_type = bin_op['rightType']
            left_operand = bin_op['left']
            right_operand = bin_op['right']
            if left_operand == 'function':
                left_operand = 'STD:function'
            if right_operand == 'function':
                right_operand = 'STD:function'
            
            if not left_operand in model.wv:
                left_operand_OOV += 1
            if not right_operand in model.wv:
                right_operand_OOV += 1
            if not left_operand in model.wv and not right_operand in model.wv:
                both_operands_OOV += 1
            
            if left_op_type == 'unknown':
                left_op_type_unk += 1
            if right_op_type == 'unknown':
                right_op_type_unk += 1
            if left_op_type == 'unknown' and right_op_type == 'unknown':
                both_op_types_unk += 1
            if not left_operand in model.wv and not right_operand in model.wv and \
                    left_op_type == 'unknown' and right_op_type == 'unknown':
                all_op_parts_unk += 1

def analyze_call(file, model):
    global calls_with_two_args, no_base, base_OOV, callee_OOV, first_arg_OOV, second_arg_OOV, both_args_OOV
    global base_and_callee_OOV, base_and_args_OOV, callee_and_one_arg_OOV, callee_and_args_OOV, all_call_parts_OOV

    with open(file, 'r') as f:
        json_data = f.read()
        calls = json.loads(json_data)
        for call in calls:
            if len(call['arguments']) != 2:
                continue
            
            calls_with_two_args += 1
            base = call['base']
            callee = call['callee']
            arguments = call['arguments']
            if arguments[0] == 'function':
                arguments[0] = 'STD:function'
            if arguments[1] == 'function':
                arguments[1] = 'STD:function'
            
            if base == '':
                no_base += 1
            if not base in model.wv:
                base_OOV += 1
            if not callee in model.wv:
                callee_OOV += 1
            if not arguments[0] in model.wv:
                first_arg_OOV += 1
            if not arguments[1] in model.wv:
                second_arg_OOV += 1
            if not arguments[0] in model.wv and not arguments[1] in model.wv:
                both_args_OOV += 1
                # print('%s.%s(%s, %s)' % (base, callee, arguments[0], arguments[1]))
            if not base in model.wv and not callee in model.wv:
                base_and_callee_OOV += 1
            if not base in model.wv and not arguments[0] in model.wv and not arguments[1] in model.wv:
                base_and_args_OOV += 1
            if not callee in model.wv and (not arguments[0] in model.wv or not arguments[1] in model.wv):
                callee_and_one_arg_OOV += 1
            if not callee in model.wv and not arguments[0] in model.wv and not arguments[1] in model.wv:
                callee_and_args_OOV += 1
            if not base in model.wv and not callee in model.wv and not arguments[0] in model.wv \
                 and not arguments[1] in model.wv:
                all_call_parts_OOV += 1
            
            voc.add(base)
            voc.add(callee)
            voc.add(arguments[0])
            voc.add(arguments[1])


if __name__ == '__main__':
    model = Word2Vec.load(sys.argv[1])
    data_folder = sys.argv[2]
    what = sys.argv[3]
    filetype = sys.argv[4]

    for calls_file in iterate_instances(data_folder, what, filetype):
        analyze_call(calls_file, model)
    
    print('Calls with two arguments: %d' % (calls_with_two_args))
    print('No Base Object: %d %f%%' % (no_base, 100 * float(no_base) / calls_with_two_args))
    print('Base Object OOV: %d %f%%' % (base_OOV, 100 * float(base_OOV) / calls_with_two_args))
    print('Callee OOV: %d %f%%' % (callee_OOV, 100 * float(callee_OOV) / calls_with_two_args))
    print('First Argument OOV: %d %f%%' % (first_arg_OOV, 100 * float(first_arg_OOV) / calls_with_two_args))
    print('Second Argument OOV: %d %f%%' % (second_arg_OOV, 100 * float(second_arg_OOV) / calls_with_two_args))
    print('Both Arguments OOV: %d %f%%' % (both_args_OOV, 100 * float(both_args_OOV) / calls_with_two_args))
    print('Base and Callee OOV: %d %f%%' % (base_and_callee_OOV, 100 * float(base_and_callee_OOV) / calls_with_two_args))
    print('Base and Args OOV: %d %f%%' % (base_and_args_OOV, 100 * float(base_and_args_OOV) / calls_with_two_args))
    print('Callee any ARG OOV: %d %f%%' % (callee_and_one_arg_OOV, 100 * float(callee_and_one_arg_OOV) / calls_with_two_args))
    print('Callee and ARGs OOV: %d %f%%' % (callee_and_args_OOV, 100 * float(callee_and_args_OOV) / calls_with_two_args))
    print('Everything OOV: %d %f%%' % (all_call_parts_OOV, 100 * float(all_call_parts_OOV) / calls_with_two_args))

    print()
    print(len(voc))
    # for binOps_file in iterate_instances(data_folder, what, filetype):
    #     analyze_binOp(binOps_file, model)

    # print('BinOps: %d' % (bin_ops))
    # print('Left Operand OOV: %d %f%%' % (left_operand_OOV, 100 * float(left_operand_OOV) / bin_ops))
    # print('Right Operand  OOV: %d %f%%' % (right_operand_OOV, 100 * float(right_operand_OOV) / bin_ops))
    # print('Both Operands OOV: %d %f%%' % (both_operands_OOV, 100 * float(both_operands_OOV) / bin_ops))
    # print('Left Operand Type Unknown: %d %f%%' % (left_op_type_unk, 100 * float(left_op_type_unk) / bin_ops))
    # print('Right Operand Type Unknown: %d %f%%' % (right_op_type_unk, 100 * float(right_op_type_unk) / bin_ops))
    # print('Both Operand Types Unknown: %d %f%%' % (both_op_types_unk, 100 * float(both_op_types_unk) / bin_ops))
    # print('Everything OOV or Unknown: %d %f%%' % (all_op_parts_unk, 100 * float(all_op_parts_unk) / bin_ops))
