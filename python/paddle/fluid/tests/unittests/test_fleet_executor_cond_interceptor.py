# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# import unittest

import os
import re

import paddle
import paddle.fluid.core as core
from paddle.distributed.fleet.fleet_executor_utils import TaskNode

paddle.enable_static()

import paddle.distributed.fleet as fleet
from paddle.distributed.fleet.meta_optimizers.common import CollectiveHelper
from paddle.distributed.fleet.meta_optimizers.sharding_optimizer import (
    ShardingOptimizer,
)

# dist_strategy = fleet.DistributedStrategy()

fleet.init(is_collective=True)


startup_program_ = None
cond_var2 = None

name_idx = 0


class nameGen:
    @staticmethod
    def generate(name):
        global name_idx
        name_idx += 1
        return name + str(name_idx)


def cond(i, ten):
    return i < ten
    # return paddle.cast(i < ten, paddle.bool)
    # return i < ten


def body(i, ten):
    i2 = i + 1
    paddle.static.Print(i, message="i")
    paddle.static.Print(i2, message="i2")
    # paddle.static.Print(i2, message="0-1")
    j = i2 + 1
    paddle.static.Print(j, message="j")
    paddle.static.Print(j, message="1-2")
    k = j + 1
    paddle.static.Print(k, message="k")
    # paddle.static.Print(j, message="2-3")
    p = k + 1
    paddle.assign(p, i)
    paddle.static.Print(p, message="res")
    return [i, ten]


def create_var(program, var):
    program.block(0).create_var(
        name=var.name,
        shape=var.shape,
        dtype=var.dtype,
        stop_gradient=var.stop_gradient,
    )


ring_map = {}


class TestCollectiveRunnerBase(ShardingOptimizer):
    def __init__(self):
        self.global_ring_id = 1
        self._startup_program = paddle.static.default_startup_program()
        print()
        self.current_endpoint = os.environ.get("PADDLE_CURRENT_ENDPOINT")
        self.end_points = os.environ.get("PADDLE_TRAINER_ENDPOINTS").split(",")
        self.pp_rank = fleet.worker_index()
        self.pp_ring_id = 4
        # self.pipeline_pair = [
        #     (0,1), (1,2), (2,3), (3,2),
        #     (3, 1), (3, 0)
        # ]
        self.pipeline_pair = [(0, 1), (1, 0)]
        self.pp_ring_map = {}
        idx = 100
        for pair in self.pipeline_pair:
            key = pair[0] * 1000 + pair[1]
            self.pp_ring_map[key] = idx
            idx += 1
        self.pp_group_endpoints = self.end_points

        for k, v in os.environ.items():
            print(f"{k} : {v}")

        self._collective_helper = CollectiveHelper(
            fleet.util.role_maker, nrings=1
        )
        self._init_pipeline_comm(self._startup_program.block(0))

        global ring_map
        ring_map = self.pp_ring_map


def is_send_op(op):
    return True


def is_recv_op(op):
    return True


def is_cut(op):
    return op.type == "print" and "-" in op.attr("message")


def get_ring_id(src, dst):
    global ring_map
    return ring_map[src * 1000 + dst]


def insert_send_op(idx, block, var, src, dst):
    global rings
    if idx >= 0:
        block.append_op(
            index=idx,
            type="send_v2",
            inputs={'X': var},
            attrs={
                'ring_id': get_ring_id(src, dst),
                'peer': 1,
                'use_calc_stream': False,
                'dynamic_shape': True,
            },
        )
    else:
        block.append_op(
            type="send_v2",
            inputs={'X': var},
            attrs={
                'ring_id': get_ring_id(src, dst),
                'peer': 1,
                'use_calc_stream': False,
                'dynamic_shape': True,
            },
        )


def insert_recv_op(idx, block, var, src, dst):
    global rings
    if idx >= 0:
        block._insert_op(
            index=idx,
            type="recv_v2",
            outputs={'Out': var},
            attrs={
                'peer': 0,
                'ring_id': get_ring_id(src, dst),
                'dtype': core.VarDesc.VarType.INT8
                if var.dtype == core.VarDesc.VarType.BOOL
                else var.dtype,
                'out_shape': var.shape,
                'use_calc_stream': True,
                'dynamic_shape': True,
            },
        )
    else:
        block.append_op(
            type="recv_v2",
            outputs={'Out': var},
            attrs={
                'peer': 0,
                'ring_id': get_ring_id(src, dst),
                'dtype': core.VarDesc.VarType.INT8
                if var.dtype == core.VarDesc.VarType.BOOL
                else var.dtype,
                'out_shape': var.shape,
                'use_calc_stream': True,
                'dynamic_shape': True,
            },
        )


def append_op(block, op):
    block.append_op(
        type=op.type,
        inputs=op.desc.inputs(),
        outputs=op.desc.outputs(),
        attrs=op.all_attrs(),
    )


def cast2bool(var, var2, block):
    block.append_op(
        type="cast",
        inputs={"X": var},
        outputs={"Out": var2},
        attrs={
            "in_dtype": core.VarDesc.VarType.INT8,
            "out_dtype": core.VarDesc.VarType.BOOL,
        },
        stop_gradient=True,
    )
    return var2


def cast2int(var, var2, block):
    block.append_op(
        type="cast",
        inputs={"X": var},
        outputs={"Out": var2},
        attrs={
            "in_dtype": core.VarDesc.VarType.BOOL,
            "out_dtype": core.VarDesc.VarType.INT8,
        },
        stop_gradient=True,
    )


def append_last_send_op(p1, src, dst):
    last_op = p1.block(0).ops[-1]
    assert (
        len(last_op.output_names) == 1
    ), f"{p1}\n{last_op.output_names}\n{last_op.output_names[0]}\n len: {len(last_op.output_names)} \n len2: {len(last_op.output_names[0])}"
    var_name = last_op.output(last_op.output_names[0])[0]
    var = p1.block(0).var(var_name)
    insert_send_op(-1, p1.block(0), var, src, dst)


def insert_first_recv_op(p1, src, dst):
    first_op = p1.block(0).ops[0]
    # assert len(first_op.input_names) == 1
    try:
        var_name = first_op.input(first_op.input_names[1])[0]
    except:
        print(first_op.type)
        print(first_op.input_names)
        print(first_op.input(first_op.input_names[1]))
        raise RuntimeError("")
    var = p1.block(0).var(var_name)
    insert_recv_op(0, p1.block(0), var, src, dst)


def create_recv_program(broadcast_vars, p1, src, dst):
    p2 = paddle.static.Program()
    for var, workers in broadcast_vars.items():
        create_var(p2, var)
        for worker in workers:
            if src == worker:
                continue
            if dst != worker:
                continue
            if var.dtype == paddle.bool:
                create_var(p2, cond_var2)
                insert_recv_op(-1, p2.block(0), cond_var2, src, dst)
                cast2bool(cond_var2, var, p2.block(0))
                paddle.static.Print(
                    var, print_phase="forward", message="recved"
                )
            else:
                insert_recv_op(-1, p2.block(0), var, src, worker)
        # cast2bool(var_int, var, p2.block(0))
    return p2


def split_block(program, stage, num_stage, broadcast_vars):
    p1 = program.clone()
    p2 = paddle.static.Program()
    if stage == 0:
        append_last_send_op(p1, 0, 1)
        p2 = create_recv_program(broadcast_vars, p1, num_stage - 1, 0)
    elif stage < num_stage - 1:
        insert_first_recv_op(p1, stage - 1, stage)
        append_last_send_op(p1, stage, stage + 1)
        p2 = create_recv_program(broadcast_vars, p1, num_stage - 1, stage)
    else:
        insert_first_recv_op(p1, stage - 1, stage)
        for var, workers in broadcast_vars.items():
            for worker in workers:
                if stage != worker:
                    insert_send_op(-1, p1.block(0), var, stage, worker)
    return p1, p2


def split_body(main_program, stage, num_stage, broadcast_vars):
    program_b_0 = paddle.static.Program()
    program_b_1 = paddle.static.Program()
    for var_name in main_program.block(0).vars:
        if re.search(".*_generated_var_.*", var_name):
            continue
        create_var(program_b_0, main_program.block(0).var(var_name))
        create_var(program_b_1, main_program.block(0).var(var_name))
    for var_name in main_program.block(1).vars:
        if re.search(".*_generated_var_.*", var_name):
            continue
        create_var(program_b_0, main_program.block(1).var(var_name))
        create_var(program_b_1, main_program.block(1).var(var_name))
    if stage < num_stage - 1:
        program_b_1 = paddle.static.Program()

    cut = 0
    for op in main_program.block(1).ops:
        if is_cut(op):
            cut += 1
            continue
        if cut == stage:
            append_op(program_b_0.block(0), op)
    p1, p2 = split_block(program_b_0, stage, num_stage, broadcast_vars)
    return p1, p2


class TestFleetExecutor:
    def test_cond_interceptor(self):
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        global startup_program_
        startup_program_ = startup_program
        with paddle.static.program_guard(main_program, startup_program):
            with paddle.static.program_guard(startup_program, None):
                counter = paddle.full(shape=[1], fill_value=0, dtype='int64')
                counter.persistable = True
            main_counter = main_program.block(0).create_var(
                name=counter.name,
                shape=counter.shape,
                dtype=counter.dtype,
                stop_gradient=counter.stop_gradient,
                persistable=counter.persistable,
            )

            i = main_counter * 1000
            paddle.assign(main_counter + 1, main_counter)
            i.persistable = False
            ten = i + 10
            ten.persistable = False
            global cond_var2
            cond_var2 = paddle.full(
                shape=[1], fill_value=0, dtype=paddle.int32
            ).astype("int8")
            i, ten, cond_var = paddle.static.nn.while_loop(cond, body, [i, ten])
            paddle.static.Print(i, message="result")
            common = TestCollectiveRunnerBase()

        program_a = paddle.static.Program()
        program_b_0 = paddle.static.Program()
        program_b_1 = paddle.static.Program()
        program_c = paddle.static.Program()

        print("main program")
        print(main_program)
        # block a
        for var_name in main_program.block(0).vars:
            if re.search(".*_generated_var_.*", var_name):
                continue
            var = main_program.block(0).var(var_name)
            program_a.block(0).create_var(
                name=var_name,
                shape=var.shape,
                dtype=var.dtype,
                stop_gradient=var.stop_gradient,
            )
            program_c.block(0).create_var(
                name=var_name,
                shape=var.shape,
                dtype=var.dtype,
                stop_gradient=var.stop_gradient,
            )
        set_a = True
        for op in main_program.block(0).ops:
            if op.type != "while":
                if set_a:
                    program_a.block(0).append_op(
                        type=op.type,
                        inputs=op.desc.inputs(),
                        outputs=op.desc.outputs(),
                        attrs=op.all_attrs(),
                    )
                else:
                    program_c.block(0).append_op(
                        type=op.type,
                        inputs=op.desc.inputs(),
                        outputs=op.desc.outputs(),
                        attrs=op.all_attrs(),
                    )
            else:
                set_a = False
        broadcast_vars = {cond_var: [0, 1], i: [0, 1]}

        # block_b1
        p01, p02 = split_body(
            main_program,
            fleet.worker_index(),
            fleet.worker_num(),
            broadcast_vars,
        )
        # p11, p12 = split_body(main_program, 1, 4, broadcast_vars)
        # p21, p22 = split_body(main_program, 2, 4, broadcast_vars)
        # p31, p32 = split_body(main_program, 3, 4, broadcast_vars)

        for stage in "0":
            for block in "12":
                p = eval(f"p{stage}{block}")
                print(f"stage: {stage} block: {block}")
                print(p)
        print("main")
        print(main_program)
        print("start")
        print(startup_program)
        # return
        cond_var_name = cond_var.name
        print("cond var name:", cond_var_name)
        num_micro_batches = 5
        rank = fleet.worker_index()
        task_id = fleet.worker_index() * 1000
        task_a = TaskNode(
            rank,
            num_micro_batches,
            0,
            node_type="Compute",
            task_id=0 + task_id,
            program=program_a,
            lazy_initialize=True,
        )
        task_b = TaskNode(
            rank,
            num_micro_batches,
            0,
            node_type="Cond",
            task_id=1 + task_id,
            program=paddle.static.Program(),
            cond_var_name=cond_var_name,
            lazy_initialize=True,
        )
        task_c = TaskNode(
            rank,
            num_micro_batches,
            0,
            node_type="Compute",
            task_id=2 + task_id,
            program=p01,
            lazy_initialize=True,
        )
        task_d = TaskNode(
            rank,
            num_micro_batches,
            0,
            node_type="Compute",
            task_id=3 + task_id,
            program=p02,
            lazy_initialize=True,
        )
        task_e = TaskNode(
            rank,
            num_micro_batches,
            0,
            node_type="Compute",
            task_id=4 + task_id,
            program=program_c,
            lazy_initialize=True,
        )

        task_a.add_downstream_task(task_b.task_id(), 2)
        task_b.add_upstream_task(task_a.task_id(), 2)
        task_b.add_downstream_task(task_c.task_id(), 100)
        task_c.add_upstream_task(task_b.task_id(), 100)
        task_c.add_downstream_task(task_d.task_id(), 2)
        task_d.add_upstream_task(task_c.task_id(), 2)
        task_d.add_downstream_task(task_b.task_id(), 100, core.DependType.LOOP)
        task_b.add_upstream_task(task_d.task_id(), 100, core.DependType.LOOP)
        task_b.add_downstream_task(
            task_e.task_id(), 100, core.DependType.STOP_LOOP
        )
        task_e.add_upstream_task(
            task_b.task_id(), 100, core.DependType.STOP_LOOP
        )
        id_to_rank = {
            task_a.task_id(): rank,
            task_b.task_id(): rank,
            task_c.task_id(): rank,
            task_d.task_id(): rank,
            task_e.task_id(): rank,
        }
        print(f"id to rank: {id_to_rank}")
        main_program._pipeline_opt = {
            "fleet_opt": {
                'tasks': [task_a, task_b, task_c, task_d, task_e],
                'task_id_to_rank': {
                    task_a.task_id(): rank,
                    task_b.task_id(): rank,
                    task_c.task_id(): rank,
                    task_d.task_id(): rank,
                    task_e.task_id(): rank,
                },
                'num_micro_batches': num_micro_batches,
            },
        }

        place = paddle.fluid.CUDAPlace(fleet.worker_index())
        exe = paddle.fluid.Executor(place)
        exe.run(startup_program)
        print("run main")
        exe.run(main_program)


if __name__ == "__main__":
    test = TestFleetExecutor()
    test.test_cond_interceptor()
    # unittest.main()
