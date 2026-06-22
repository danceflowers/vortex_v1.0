SIM ?= iverilog
VVP ?= vvp
TOP ?= tb_tensor_core_sparse
OUT ?= simv

RTL = \
	define.v \
	fp_to_fp9.v \
	fp_to_fp22.v \
	input_parser_bypass.v \
	matrix_input_buffer.v \
	matrix_input_buffer_sparse.v \
	to_intype_con.v \
	to_next_con.v \
	spar_mul.v \
	tc_mul_dual_path.v \
	mm_mul_add.v \
	to_outtype_con.v \
	tensor_core.v \
	tb_tensor_core_sparse.v

all: $(OUT)

$(OUT): $(RTL)
	$(SIM) -g2012 -Wall -s $(TOP) -o $(OUT) $(RTL)

run: $(OUT)
	$(VVP) $(OUT)

clean:
	rm -f $(OUT) *.vcd

