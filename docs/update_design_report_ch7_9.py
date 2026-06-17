from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W}
ET.register_namespace("w", W)

DOCX_PATH = Path("docs/OpenTensorCore1+Vortex C Model设计集成报告.docx")
DEFAULT_OUTPUT_PATH = Path("docs/OpenTensorCore1+Vortex C Model设计集成报告.updated.docx")


def w(tag: str) -> str:
    return f"{{{W}}}{tag}"


def clear_paragraph_keep_ppr(p: ET.Element) -> ET.Element:
    ppr = p.find(w("pPr"))
    for child in list(p):
        p.remove(child)
    if ppr is not None:
        p.append(copy.deepcopy(ppr))
    return p


def make_run(text: str, color: str | None = None, bold: bool = False) -> ET.Element:
    r = ET.Element(w("r"))
    if color is not None or bold:
        rpr = ET.SubElement(r, w("rPr"))
        if color is not None:
            ET.SubElement(rpr, w("color"), {w("val"): color})
        if bold:
            ET.SubElement(rpr, w("b"))
    t = ET.SubElement(r, w("t"))
    if text.startswith(" ") or text.endswith(" "):
        t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    t.text = text
    return r


def make_para_from_template(template: ET.Element,
                            text: str,
                            color: str | None = None,
                            bold: bool = False) -> ET.Element:
    p = clear_paragraph_keep_ppr(copy.deepcopy(template))
    p.append(make_run(text, color=color, bold=bold))
    return p


def blank_para(template: ET.Element) -> ET.Element:
    return make_para_from_template(template, "")


def build_blocks(body_children: list[ET.Element]) -> list[ET.Element]:
    chapter_tpl = body_children[1053]
    section_tpl = body_children[1057]
    subsection_tpl = body_children[1063]
    normal_tpl = body_children[1054]

    image_arch = copy.deepcopy(body_children[1058])
    image_flow_1 = copy.deepcopy(body_children[1060])
    image_flow_2 = copy.deepcopy(body_children[1061])
    image_sparse_pipe = copy.deepcopy(body_children[1095])
    image_storage = copy.deepcopy(body_children[1720])
    image_caccum = copy.deepcopy(body_children[1773])
    image_slot_1 = copy.deepcopy(body_children[1780])
    image_slot_2 = copy.deepcopy(body_children[1784])

    blocks: list[ET.Element] = []

    def chapter(text: str) -> None:
        blocks.append(make_para_from_template(chapter_tpl, text))

    def section(text: str) -> None:
        blocks.append(make_para_from_template(section_tpl, text))

    def subsection(text: str) -> None:
        blocks.append(make_para_from_template(subsection_tpl, text))

    def normal(text: str = "") -> None:
        blocks.append(make_para_from_template(normal_tpl, text))

    def yellow(text: str) -> None:
        blocks.append(make_para_from_template(normal_tpl, text, color="FFFF00", bold=True))

    def blank() -> None:
        blocks.append(blank_para(normal_tpl))

    chapter("7 第二代vortex核心修改")
    normal("代码路径：")
    normal("本章按照当前 sim/simx Cmodel 的实际实现，对第二代 Vortex 核心中引入 OpenTensorCore、TMEM、TMA、window planner、异步 tensor control 的设计集成方式进行统一说明。")
    normal("与前一版文档相比，本章重点修正以下几类已经过时的描述：第一，TMEM 不再以“bank 级资源 + 固定 payload/meta region”作为唯一抽象，而是同时区分数学矩阵视图、TMEM 逻辑视图和物理 bank 视图；第二，MMA_LOAD/MMA_STORE/WMMA/TMA_* 的语义全部围绕 descriptor、window、tile、packet 重新统一；第三，C 累加域已经改为 CMem(fp22) 直接累加，不再依赖独立 accumulator。")
    normal("当前代码主路径位于：sim/simx/core.cpp、sim/simx/tmem.cpp、sim/simx/tmem_window_planner.cpp、sim/simx/tensor_unit.cpp、sim/simx/execute.cpp、sim/simx/decode.cpp，以及 kernel/include/vx_tensor.h。")

    section("7.1 整体架构图")
    yellow("【黄字建议补图：第二代 simx tensor 路径整体架构图。建议图中画出 Host/Test、descriptor table、Core、TMEM、TensorUnit、OpenTensorCore、AMem/BMem/CMem/MetaMem 的连接关系，并标出 TMA_LOAD、MMA_LOAD、WMMA、MMA_STORE、TMEM_SHIFT 五条主链。】")
    blocks.append(image_arch)
    normal("当前整体架构可以分为三层视图。第一层是数学矩阵视图，A/B/C 的 shape 表示元素维度，例如 A_shape=16x64 表示 16 行 64 列的数学矩阵；第二层是 TMEM 逻辑视图，window planner 根据矩阵 shape、fmt、sparse mode 和 allocation 大小生成 window footprint；第三层是物理 bank 视图，bank mapper 再把逻辑地址映射到多 bank SRAM，尽量把 512bit packet 打散到不同 bank 上，以提高带宽利用率。")
    normal("执行主线为：Host/Test 构造 descriptor 和数据 -> Core 读 descriptor -> TMA 将一个 window 的数据搬运到 TMEM -> MMA_LOAD 按 window_id 与 tile_id 从 TMEM 读取当前 tile 的 packet 流并填充本地存储 -> WMMA 以宏指令形式入队，在 TensorUnit 内拆成 8 条 primitive uop -> primitive 结果直接在 CMem(fp22) 中累计 -> MMA_STORE 按 tile 顺序将结果窗口写回 TMEM -> TMA_STORE 再将结果从 TMEM 写回外部内存。")
    normal("这条链路的关键点是：数学矩阵语义只在 planner 和 MMA/TMA packet 生成阶段解释一次；之后所有搬运与仲裁都围绕逻辑 window 和物理 packet 进行。这样既保留了矩阵乘法的语义正确性，又避免让软件显式感知 bank 布局。")
    normal("执行流程：")
    yellow("【黄字建议补图：数据执行流程图。建议图一画“descriptor -> window planner -> TMA -> TMEM -> MMA_LOAD -> TensorUnit -> WMMA -> CMem(fp22) -> MMA_STORE -> TMA_STORE”的顺序；建议图二画“控制流：TC_COMMIT / MBAR / TC_FENCE / TC_WAIT”的同步关系。】")
    blocks.append(image_flow_1)
    blocks.append(image_flow_2)
    normal("在当前实现中，Core::tick() 仍然是全局唯一时钟推进入口。所有 tensor 相关异步事件，包括 TMA_LOAD、TMA_STORE、TMEM_SHIFT、MMA_LOAD、MMA_STORE 和 WMMA，最终都在这条主时序下被推进，不引入第二套 event queue 或第二个 clock domain。")

    section("7.2  2:4和1:4稀疏化")
    subsection("7.2.1 稀疏化各模块说明")
    normal("本节保留原有稀疏化模块划分，但对其在当前 Cmodel 中的角色做更清晰的说明。当前主线仍以 dense path 收敛为第一优先级，稀疏 path 采取“保持接口、分步接入、先保证 metadata 与 payload 的搬运路径一致，再逐步恢复计算侧语义”的策略。")
    normal("1）input_parser_bypass")
    normal("功能：接收来自精度转换模块前后的权重输入、metadata 以及 sparse_mode 相关控制信号。当检测到当前输入已经是预压缩格式时，模块直接旁路压缩后的 A payload 与 metadata，不再做额外的入口压缩。这样可以减少前端重复解析，并保证上层对 payload/meta 的打包方式有完全控制。")
    normal("2）spar_mul 模块（原 sparse_router）")
    normal("功能：在稀疏路径中完成“metadata 解析 + B 元素抽取 + 稀疏 lane 激活”。对于 2:4 模式，每个 4 元素 group 激活 2 条有效乘法 lane；对于 1:4 模式，每个 group 激活 1 条有效 lane。空闲 lane 输出零值部分积，使其能够继续接入统一的后级加法树。")
    normal("3）tc_mul_dual_path 模块")
    normal("功能：保留 Dense 原始部分积路径，并在 sparse_mode 非零时切换到 spar_mul 输出。这保证 dense path 的关键路径不会因为稀疏模块而增加组合延迟，同时也便于在验证阶段分别对 dense/sparse 两条路径做对拍。")
    normal("4）output_packer 模块")
    normal("功能：当前版本主要承担接口占位和输出格式整理职责，后续可扩展为输出压缩、metadata 回写或 block scaling 相关格式包装。本版 Cmodel 重点验证其接口位置与后端结果组织是否正确。")
    normal("5）预留模块")
    normal("bmem_sparse_perceptor、router_deep_fold 等预留模块目前主要用于接口占位、端口连通性和未来扩展点预留，不作为本版收敛主路径的一部分。")

    subsection("7.2.2 电路结构与流水线")
    yellow("【黄字建议补图：稀疏路径流水线示意图。建议包含 input_parser_bypass、metadata 旁路、spar_mul、dual-path mux、后级加法树，并标注 Dense Path 与 Sparse Path 的切换位置。】")
    blocks.append(image_sparse_pipe)
    normal("稀疏化电路跨越输入准备、乘法前端和加法规约三个阶段，但不改变原张量核主体的主框架。")
    normal("第一阶段：入口标志位解析与旁路。输入数据进入 AMem 前，首先解析 sparse_mode、压缩标志和 metadata 是否有效。如果当前数据已经由外部预压缩完成，则 payload 与 metadata 直接按既定格式写入本地存储，而不是在入口再次做零值统计或压缩。")
    normal("第二阶段：稀疏乘法前端。稀疏前端从 A payload 与 metadata 中解析每个 group 的真实非零位置，再根据这些位置从 B 中抽取对应元素，形成统一的稀疏部分积。")
    normal("第三阶段：双路径选择与后级规约。tc_mul_dual_path 将 dense 路径与 sparse 路径输出统一后再送入后级。Dense 模式保持原生阵列路径；Sparse 模式则使用 spar_mul 的结果。这样可以保证 dense path 100% 向后兼容，并尽量把 sparse 功能的侵入范围限制在前端与选择层。")

    subsection("7.2.3   1:4、2:4稀疏化在tcgen05指令中的配置")
    subsection("7.2.3.1 压缩格式")
    normal("2:4 模式下，每 4 个元素保留 2 个非零值，因此每个 4 元组需要 2 个 payload value 和 4bit metadata。1:4 模式下，每个 4 元组只保留 1 个非零值，因此只需要 1 个 payload value 和 2bit metadata。")
    normal("若 A tile 为 16x16，按 K 方向每 4 个元素分组，则一共存在 64 个 group。以 fp16 为例，dense 原始数据大小为 512B；2:4 压缩后 payload 为 256B，metadata 为 32B，总共 288B；1:4 压缩后 payload 为 128B，metadata 为 16B，总共 144B。以 fp8 为例，dense 为 256B；2:4 压缩后为 160B；1:4 压缩后为 80B。")
    normal("从软件打包和硬件解包角度看，payload 与 metadata 分离仍然是更合理的组织方式。这样在 TMA 搬运、TMEM 存放、MMA_LOAD 解包、MetaMem 写入各阶段都更清晰，也更容易单独验证 payload 与 metadata 的有效性。")

    subsection("7.2.3.2 TMEM如何存放")
    normal("原始设计中，TMEM 从逻辑上区分 payload region 与 meta region，这一思路在当前版本仍然保留其语义价值：payload 与 metadata 不是同一种资源，不应在软件层强行交织。")
    normal("但在当前 Cmodel 中，这一设计已经进一步演化。TMEM 不再使用“固定 payload 区 + 固定 meta 区”的唯一表达，而是统一抽象为 window。对于 dense A/B/C，window 直接表示 payload window；对于 sparse A，则除了 payload window 外，还会派生一个内部的 meta shadow window，用来保存与当前 A tile 一一对应的 metadata packet。")
    normal("因此，metadata 仍然与 payload 逻辑分离，但它们共享同一套 TMEM 物理 bank 阵列，只是在 window planner 和 layout resolver 中被映射到不同的逻辑 window 区域。这样可以保留“metadata 是独立操作数”的编程语义，同时又避免在物理实现上额外增加一套独立 SRAM。")
    normal("对一个 16x16 的 A tile 而言，当前设计采用“先完整搬运 A payload，再一次性搬运 1 个 64B meta packet”的方式。这个 64B meta packet 在进入本地 MetaMem 后，再逻辑拆成 4 条 16B meta line，分别对应 4 个 8x8 A block 的使用需求。")

    subsection("7.2.3.3 descriptor配置")
    normal("在当前实现中，descriptor 不再只是“几项格式字段”的容器，而是整套 window/tile/packet 语义的入口。")
    normal("TMA descriptor 主要服务 TMA_LOAD、TMA_STORE、TMEM_SHIFT refill，核心字段包括：外部内存地址、rows、cols、stride_bytes、elem_bytes、flags、tile_role、payload_kind，以及可选的 metadata 地址与 metadata 大小。")
    normal("MMA descriptor 主要服务 MMA_LOAD、MMA_STORE、WMMA，核心字段包括：fmt_a、fmt_b、fmt_c、fmt_d、ws、sp、sparse_mode、transpose_a、transpose_b，以及 A/B/C 的数学矩阵 shape。这里的 A_shape/B_shape/C_shape 表示的是数学矩阵元素维度，而不是 TMEM 逻辑 footprint。")
    normal("window planner 会以 MMA descriptor 为输入，将数学矩阵 shape、fmt 和 sparse_mode 进一步翻译为 TMEM 中的 logical_col_span、logical_line_span、tile_count、packets_per_tile 等布局信息。")

    section("7.3 非对称FP8/FP16输入")
    normal("对于非对称 FP8/FP16 输入，当前设计不再把 fmt_ab 强行绑定为同一精度，而是在 MMA descriptor 中分别记录 fmt_a 和 fmt_b，并允许 fmt_c、fmt_d 独立存在。")
    normal("这样做的直接收益有两个。第一，A/B 两路可以独立采用最合适的输入精度；第二，本地填充路径与 OpenTensorCore 的精度转换逻辑可以解耦。当前 Cmodel 中，fp8/fp16 到 fp9 的转换发生在写入 AMem/BMem 的本地 beat 写入阶段，fp16/fp32 到 fp22 的转换发生在写入 CMem 之前。")
    normal("因此，从软件视角看，MMA_LOAD 只需要显式指定 target、window_id、tile_id 和 slot_id；从执行视角看，真正的数据类型转换发生在 TensorUnit 的 fill path，而不是由 decode 直接展开为多条不同精度的微指令。")

    section("7.4 tcgen05 指令兼容")
    subsection("7.4.1 tcgen05 指令内容")
    normal("tcgen05 是 Blackwell 架构引入的面向 Tensor Memory 与异步矩阵计算的指令前缀。其核心思想不是简单增加几条矩阵指令，而是引入一套“显式内存管理 + 显式搬运 + 异步提交与等待 + 软件管理的流水控制”体系。")
    normal("当前 Vortex Cmodel 与 tcgen05 的对齐重点不在于逐位复刻 PTX 语法，而在于复刻其核心语义：TMEM 按列分配，数据搬运走异步 TMA，矩阵计算以单线程触发的异步 WMMA 为中心，同步通过 commit/wait/fence/mbarrier 完成。")

    subsection("7.4.2 修改方法")
    normal("原来的 TMEM_ALLOC / TMEM_FREE / TMA_LOAD / TMA_STORE / TMA_WAIT / MMA_LOAD / MMA_STORE / WMMA 构成了资源管理和数据搬运的基础，但要对齐 tcgen05，还需要控制层语义，包括 allocation permit、commit domain、fence、shift 以及 mbarrier。")
    normal("当前版本的处理方式是：保留原有 RISC-V custom 指令风格，将复杂、低频、但对全局语义关键的字段放到 descriptor 或 control word 中；同时把矩阵计算与数据搬运统一到一套 window/tile/packet 语义下。")

    subsection("7.4.2.1 新增指令")
    normal("1）TMEM_REL_PERMIT")
    normal("语义：将当前 core 的 TMEM allocator 状态置为 sealed。之后新的 TMEM_ALLOC 将被拒绝，直到 kernel 结束或外部重置。该指令用于模拟 tcgen05 中对 allocation permit 的控制。")
    normal("2）TC_COMMIT")
    normal("语义：把本线程此前已发出的异步 tensor 操作挂接到某个 barrier 或 commit domain。当前实现追踪的对象包括 TMA_LOAD、TMA_STORE、TMEM_SHIFT 以及异步 WMMA。TC_COMMIT 并不直接等待这些操作完成，而是把它们登记到 barrier 的 pending_tx 中。")
    normal("3）TC_FENCE")
    normal("语义：提供 BEFORE/AFTER 两种模式。BEFORE 用于保证此前异步 tensor 操作在同步点之前完成排序；AFTER 用于保证同步点之后才去消费已经完成的 tensor 结果。")
    normal("4）TC_WAIT")
    normal("语义：等待本 warp 上已经登记的本地异步 tensor 事件收敛。该指令用于补足“本 warp 自身等待”语义，而跨 warp 仍然使用 TC_COMMIT + MBAR_ARRIVE + MBAR_WAIT。")
    normal("5）TMEM_SHIFT")
    normal("语义：外部语义已经从“按逻辑 tile 粗粒度下移”进一步明确为“按数学矩阵行向下移动一行”。内部实现仍然允许通过逻辑 line 与 packet 重构来完成这一动作，从而兼容未来更复杂的逻辑布局。")
    normal("6）MBAR_INIT / MBAR_ARRIVE / MBAR_WAIT")
    normal("语义：维护 barrier object 的 phase、pending_arrivals、pending_tx 和 waiters_bitmap。它是当前 tcgen05 风格异步提交与等待链路中的核心同步原语。")

    subsection("7.4.2.2 RISC-V 指令兼容方法")
    normal("当前 Cmodel 继续使用 custom-0 major opcode，并通过 funct3/funct7 区分 TMEM、TMA、MMA、WMMA 与控制类指令。")
    normal("与旧版文档不同，当前实现已经不再依赖 tagged handle 传递 window 信息。TMA_LOAD、TMA_STORE、TMEM_SHIFT 的 window_id、descriptor_id、flags 统一编码在 rs2 control 字段中；MMA_LOAD、MMA_STORE 的 target、slot_id、window_id、tile_id 也统一编码在 rs2 control 字段中。")
    normal("这种控制字设计的好处是：handle 继续只表达 allocation 资源，不再兼任 window 或 tile 语义；软件能够显式指定 window_id 和 tile_id；decode 仍然只负责静态字段解析，真正的 runtime handle、window、tile 选择留到 execute 阶段处理。")
    normal("旧版 non-descriptor 的 MMA_LOAD/MMA_STORE 直通路径已经不再作为主路径保留，当前软件接口和 decode 均围绕 descriptor + control word 的方式展开，这样后续扩展 fmt、shape、window planner 和 sparse mode 时更容易维护。")

    subsection("7.4.2.3 TMEM Allocator State 和inflight tracker")
    normal("为了让硬件模型清楚区分“还能不能 alloc TMEM”和“哪些异步 tensor op 已经 commit 但还未完成”，当前 Core 内部维护了 allocator state、allocation table 与 inflight tracker。")
    normal("TMEM allocation 的粒度已经从旧文档里的 bank_span 语义演化为 col_span 语义。软件侧通过 handle 看到的是一段连续 logical columns；底层再把这些逻辑列映射到物理 bank。")
    normal("每个 allocation 除了 valid、payload_col_base、col_span 之外，还会记录 layout_valid、layout_epoch、payload_ready、meta_ready 以及当前绑定的 window plan。这样 handle 已经不再只是“起始位置 + 大小”，而是一个完整的 window 容器。")
    normal("inflight tracker 则负责追踪异步 TMA、TMEM_SHIFT、WMMA 等事件是否已经发出、是否已经 commit、是否已经完成，以及这些事件属于哪个 barrier/token。")

    subsection("7.4.2.4 mbarriar同步机制")
    normal("当前 barrier object 表项至少包含：valid、phase、expected_arrivals、pending_arrivals、pending_tx 和 waiters_bitmap。")
    normal("MBAR_INIT 用于初始化 barrier；MBAR_ARRIVE 负责减少 pending_arrivals；TC_COMMIT 用于把异步 tensor 事务登记到 pending_tx；异步事件真正完成时，finalize_async_tensor_op() 再把 pending_tx 减回去。")
    normal("因此，phase 完成条件已经明确为：pending_arrivals == 0 且 pending_tx == 0。只有两者同时满足，barrier phase 才结束，等待该 barrier 的 warp 才会被唤醒。")
    normal("这套机制直接接到当前的 warp scheduler 上，所以 barrier wait 自然表现为一种 stall reason，而不是在 execute 后用软重试去模拟。")

    subsection("7.4.2.5 warp scheduler 的修改")
    normal("warp scheduler 的核心修改方向是 tensor-aware ready 选择。传统 round-robin 只看 warp 是否 active，但对于 tensor 宏指令来说，一个 warp 即使还没被 barrier 挂起，也可能因为 handle 未 ready、slot 未 ready、payload/meta 未到齐或 tensorcore 前端不 ready 而在本拍必然 retry。")
    normal("因此，当前调度会优先过滤掉“头指令虽然是 tensor 指令，但本拍一定推进不了”的 warp，并对真正能推进 tensor 后端的 warp 提升分数。")
    normal("在当前实现里，典型优先级是：能发 WMMA 的 warp 分数最高；能完成 MMA_LOAD 的 warp 次之；能发 MMA_STORE 的 warp 再次之；普通指令或非 tensor 指令保持基础分。这样做的目的不是做复杂 lookahead，而是避免把前端发射机会浪费在明知会 retry 的 warp 上。")

    subsection("7.4.2.6 新增TMEM handle scoreboard / handle table")
    normal("TMEM handle scoreboard / handle table 不等价于经典寄存器 scoreboard，它更像一个 resource/state tracker。")
    normal("它需要跟踪：一个异步搬运对应的数据块、其所在 handle、payload/meta 的 ready 状态、是否正被 TMA_LOAD/TMA_STORE/TMEM_SHIFT 占用、是否仍被某个 slot/warp 引用。")
    normal("当前版本中，Core 会为每个 handle 维护 payload_ready 和 meta_ready。TMA_LOAD 发出时清掉 ready，完成时重新置 ready；TMA_STORE 和 TMEM_SHIFT 在进行中会把 handle 视为 busy；warp scheduler 在给 MMA_LOAD/MMA_STORE 打分时，会先查这层 handle-ready 状态。")
    normal("这样一来，MMA_LOAD_A(dense) 至少要求 A_handle.payload_ready；MMA_LOAD_A(sparse) 还额外要求 metadata 对应的 meta shadow window 已就绪；MMA_LOAD_B 与 MMA_LOAD_C 则分别要求自己的 payload ready。")

    subsection("7.4.2.6 descriptor描述符")
    normal("当前 descriptor 分为两类。")
    normal("第一类是 MMA descriptor，服务于 MMA_LOAD、MMA_STORE、WMMA。除 fmt_a/fmt_b/fmt_c/fmt_d、ws、sp、sparse_mode 外，还显式携带 A/B/C 数学矩阵 shape，以及 transpose_a、transpose_b。")
    normal("第二类是 TMA descriptor，服务于 TMA_LOAD、TMA_STORE 与 TMEM_SHIFT refill。除 addr、rows、cols、stride_bytes、elem_bytes、flags 外，还携带 tile_role、payload_kind，以及 metadata 相关地址与大小。")
    normal("descriptor table 仍然放在 Core 内部，以 desc_id 方式索引。后续所有 window planner、layout 绑定、MMA/TMA packet 生成，都以 descriptor table 为入口。")

    section("7.5 异步opentensorcore的修改")
    subsection("7.5.1 WMMA decode修改")
    normal("当前 WMMA 不再在 decode 侧展开成 8 条显式微指令，而是保留为 1 条宏指令。decode 只负责识别其为异步 tensor 事件，并把 desc_id、slot 控制字段和必要的 runtime operand 交给 execute。")
    normal("真正的 8 条 primitive uop 拆分发生在 TensorUnit 内部。一个 WMMA 宏事件会被表示为一个 job，随后由 TensorUnit 的 uop scheduler 按固定顺序拆成 8 条 8x8x8 primitive uop。")
    normal("这 8 条 primitive uop 不靠“返回顺序猜归属”，而是靠 sideband metadata 明确归属。每一级流水都携带 wid、async_id、slot_id、c_subtile_id、k_slice_id 等元信息，直到尾部 retire 时再依据这些 metadata 回写到正确的 C subtile。")

    subsection("7.5.2 tensor unit 的存储组织修改")
    yellow("【黄字建议补图：TensorUnit 存储组织图。建议图中画出两个逻辑 slot、共享的 AMem/BMem/CMem/MetaMem 物理阵列、slot0 对应 depth0..3、slot1 对应 depth4..7，以及 job queue / mem uop queue / tensorcore pipeline 的关系。】")
    blocks.append(image_storage)
    normal("当前 TensorUnit 仍然保留两个逻辑 slot，用于 ping-pong buffer 与跨 warp overlap；但底层本地存储已经从“两套对象 + 各自内部 ping-pong”收敛成“共享物理存储 + slot 地址窗口”。")
    normal("AMem、BMem、CMem、MetaMem 的物理阵列由 TensorUnit::Impl 统一持有。slot0 固定映射到低半部深度，slot1 固定映射到高半部深度。这样既能保持 slot 语义稳定，又减少两套独立 SRAM 带来的控制开销。")
    normal("TensorUnit 内部除了 local memory 之外，还维护 pending_wmma_jobs、active_wmma_job、pending_wmma_uops、pending_mem_ops 等队列与状态，用于驱动异步 MMA_LOAD、WMMA、MMA_STORE 的逐拍推进。")

    subsection("7.5.2.1 MMA_LOAD/MMA_STORE语义修改")
    normal("MMA_LOAD 不再默认等价于“同时装 A/B/C 三个操作数”，而是必须显式指定 target=A/B/C。这样才能支持 B 驻留、C 驻留、window 复用和 K phase 累加。")
    normal("在当前 ISA/Cmodel 里，MMA_LOAD/MMA_STORE 还显式携带 window_id、tile_id 和 slot_id。其中 tile_id 是软件对外显式可控的，因为当前希望让软件和测试能够明确指定当前操作哪个数学 tile；packet_id 则保留在 TensorUnit 内部推进。")
    normal("MMA_LOAD 的执行顺序是：先按 window_id + tile_id 通过 layout resolver 找到当前 tile 在 TMEM 中对应的 packet 流，再按 target 的本地映射关系把 packet 逐步写入 AMem、BMem、CMem 或 MetaMem。")
    normal("MMA_STORE 的执行顺序则相反：先按 tile_id 从 CMem 中读取当前 tile 对应的 subtile/packet，再通过 window layout 写回到 TMEM 的对应窗口位置。")

    subsection("7.5.2.2 C矩阵的累加方式")
    yellow("【黄字建议补图：CMem(fp22) 累加与回写示意图。建议图中画出 primitive result -> fp22 accumulate -> CMem line/subtile -> MMA_STORE -> TMEM 的路径。】")
    blocks.append(image_caccum)
    normal("原始设计中，C 结果先进入独立 accumulator，再在合适时机写回 CMem。当前 Cmodel 已进一步改为：CMem 直接以 fp22 作为存储域，primitive retire 时直接对 CMem 做 read-modify-write 式累加。")
    normal("这一改动带来的直接收益是：C 的可见性与最终值不再需要在 accumulator 与 CMem 之间反复同步；slot 的 c_dirty、cmem_final_valid 和 c_wmma_inflight 都能围绕单一的 CMem 状态来表达。")
    normal("对一条 16x16x16 的 WMMA 宏指令而言，TensorUnit 仍然会拆成 8 条 primitive uop。每条 primitive 对应 1 个 c_subtile_id 与 1 个 k_slice_id，retire 时按 c_subtile_id 找到对应的 8x8 subtile，在 fp22 域中做逐元素累加。")

    subsection("7.5.2.3 2 个 operand slot（amem、bmem、cmem）")
    yellow("【黄字建议补图：双 slot overlap 场景图。建议图一画 slot0 compute + slot1 load；图二画 slot0 C 驻留、slot1 A/B 交替装载；图三画 AB slot 与 C slot 解绑后的流水重叠。】")
    blocks.append(image_slot_1)
    blocks.append(image_slot_2)
    normal("两个 operand slot 的价值，不是为了让两个 warp 同时往同一块本地存储写，而是为了把“当前计算使用的 slot”和“下一块数据装载使用的 slot”隔离开。")
    normal("因此，当前模型允许 warp0 在 slot0 上持续发 WMMA，同时 warp1 或同一 warp 在 slot1 上完成下一次 MMA_LOAD。MMA_STORE 也作为异步事件存在，只有当对应 C slot 的 4 个 C subtile 都已经最终有效，且没有挂着未完成的 WMMA/store 时，才允许启动写回。")
    normal("进一步地，当前实现已经把 AB slot 与 C slot 的复用条件显式分开。A/B 更偏向 transient，只要该 slot 上前一条 WMMA 的 8 条 primitive 都已经发射进入流水线，就允许新的 MMA_LOAD_A/B 覆盖；C 则必须等挂在该 C slot 上的所有 WMMA 都真正 retire 完，才能发 MMA_STORE 或新的 MMA_LOAD_C。")

    section("7.5 仿真结果")
    subsection("7.5.1 mbarrier测试")
    normal("测试文件：tests/regression/tcu_mbarrier_async。")
    normal("本测试验证的重点不是单纯功能正确，而是 barrier phase 的完成条件已经从“arrivals 到齐”扩展为“arrivals 到齐且 committed async tx 完成”。Kernel 侧通过 tmem_alloc -> mbarrier_init -> tma_load -> tc_commit -> mbarrier_arrive -> mbarrier_wait -> mma_load/wmma/mma_store -> tma_store -> tc_commit -> mbarrier_wait 的顺序，验证 TMA 与 WMMA 真正接入 barrier 的完成路径。")
    normal("当前该测试已经在 simx 下通过，用于说明 mbarrier、TC_COMMIT 和异步 tensor op 之间的时序关系已经收敛到可用状态。")

    subsection("7.5.2 TMEM shift和fence测试")
    normal("测试文件：tests/regression/tcu_tmem_shift_fence 与 tests/regression/tcu_tmem_shift_refill。")
    normal("这组测试重点验证两件事：第一，TMEM_SHIFT 的外部语义已经统一为“数学行向下移动一行”；第二，TC_FENCE.BEFORE / AFTER 能够对 shift 前后的数据可见性进行排序。")
    normal("其中 refill 测试还额外验证了 top-line refill 的路径，即 shift 之后顶行既可以清零，也可以由一条 TMA descriptor 指定的新行进行回填。")

    subsection("7.5.3 TMEM测试")
    normal("测试文件：tests/regression/tcu_tmem_chain。")
    normal("该测试验证 TMEM allocation、window 绑定、TMA_LOAD、MMA_LOAD、MMA_STORE、TMA_STORE 之间的串接关系，以及 handle/table/window epoch 在多次链式操作中的一致性。")

    subsection("7.5.4 GEMM测试")
    normal("测试文件：tests/regression/sgemm_tcu_tmem。")
    normal("该测试验证当前主路径的 dense GEMM 语义，重点覆盖 descriptor table、window planner、TMEM 搬运、MMA_LOAD/MMA_STORE、WMMA 以及 CMem(fp22) 写回链路。")

    subsection("7.5.5 多warp pipeline测试")
    normal("测试文件：tests/regression/tcu_wmma_overlap。")
    normal("该测试用于观察多 warp 情况下的 load/compute overlap、slot 复用、handle ready、scheduler 评分以及 tensorcore pipeline 深度利用率。")

    subsection("7.5.6 1:4、2:4稀疏测试")
    normal("稀疏测试的重点不只是结果正确，还包括 payload 与 metadata 的打包、搬运、window 映射和 MetaMem 可见性是否与 dense path 保持统一风格。当前 dense 主链已经收敛，sparse 路径正在按“先搬运路径、再计算路径”的顺序逐步接回。")

    subsection("7.5.6 性能计数器")
    normal("为便于后续 RTL/Cmodel 对拍，本版设计继续保留 tensor 相关性能计数器，包括 TMA load/store 次数、TMEM read/write packet 计数、AMem/BMem/CMem/MetaMem 端口忙计数，以及各类 stall reason。")
    normal("这些计数器的价值不在于给出单一的总 cycles，而在于把总延迟拆成可归因的资源瓶颈，例如 TMEM egress 忙、bank 端口忙、本地 beat 端口忙、slot 不可复用、payload/meta 未 ready 等。")

    chapter("8 二次修改（按当前Cmodel实现整理）")
    normal("本章不再停留在“正在撰写中”的草稿状态，而是基于当前 simx 实现，对第二轮面向 cycle-accurate 的 Cmodel 收敛工作进行系统整理。")
    normal("本章的核心目标是说明：在不改变现有 simx 全局时序推进主线的前提下，如何把原先的粗粒度 tile-level 模型收敛成 packet-accurate、beat-accurate、primitive-accurate 的模型。")
    normal("与第 7 章偏架构意图不同，第 8 章偏实现与时序边界。")

    section("8.1 总体原则")
    normal("当前实现坚持沿用原有主线：Core::tick() -> advance_async_tensor_ops() -> commit/execute/issue/decode/fetch/schedule，以及 TensorUnit::tick() -> service_mem_ops() -> dispatch_compute_uop() -> tick_tensorcore()。")
    normal("因此，本轮修改不是另起一套时间系统，而是在现有时序入口下，把“整块完成才可见”的粗粒度状态细化为“packet/beat/line/subtile 逐步可见”的细粒度状态。")
    normal("这意味着整个系统仍然只有一个全局 cycle 计数，仍然使用现有端口预算和 ready/retry 机制，不会新增第二个 scheduler、第二个 event queue 或第二个 clock domain。")
    yellow("【黄字建议补图：全局时序推进图。建议画出 Core::tick 与 TensorUnit::tick 的调用顺序，并标出 advance_async_tensor_ops、service_mem_ops、dispatch_compute_uop、tick_tensorcore 的相对先后。】")

    section("8.2 重写 TensorUnit 的 memory-side 流水")
    normal("memory-side 流水的重写目标，是消除原有“所有 packet 都收齐后再一次性 materialize”的行为。当前模型要求 payload/meta 一旦完成一个 packet 的读取，就只能让与该 packet 直接相关的 line、beat、packet-valid 或 subtile-valid 变为可见，而不能提前让整个 tile 一次性变 ready。")
    normal("因此，TensorUnit 在 memory-side 上要明确回答以下问题：哪些 meta line 已经 resident，哪些 A/B line 已经 resident，哪些 C subtile 已经 resident，哪些 primitive 已经允许发射，以及哪些 store packet 已经可以写回 TMEM。")
    normal("在当前实现里，service_mem_ops() 每个周期只推进一个真实动作：要么完成 1 个 TMEM read packet，要么完成 1 个本地 write beat，要么完成 1 个 CMem read beat，要么完成 1 个 TMEM write packet。")

    section("8.3 AMem/BMem/CMem/MetaMem 从 bulk 接口到 beat-accurate")
    normal("AMem、BMem、CMem、MetaMem 不再只暴露 fill_tile()/dump_tile() 这种整块接口，而是显式支持按 beat 写入、按 beat 读出、按 valid bit 判断是否可读。")
    normal("AMem 的关键点是：只有与当前 primitive 对应的 line valid 后，read_primitive(step_m, step_k) 才允许成功。对于 fp16，一个 AMem line 需要两个 64B packet 才能组完；对于 fp8，一个 packet 即可组完一条逻辑 line。")
    normal("BMem 也不再等整块 tile 填满才 valid，而是按物理 line 或 line group 单独变为可读。这样可以让某些 primitive 在 B 的部分数据先到时就具备发射条件。")
    normal("MetaMem 当前仍然采用“1 个 64B packet 逻辑拆成 4 条 16B line”的方式，但读侧不再默认整包 ready，而是显式检查该 meta packet 是否已经写入。")
    normal("CMem 的关键变化是：valid 不再是单一总开关，而是要能表达更细粒度的可见性，例如 packet-valid、subtile-valid、dirty 位以及最终值是否已经稳定。")

    section("8.4 MemUop 从计数器模型到 staged 状态机")
    normal("原有 MemUop 更像一个“剩余计数器集合”，只能表达还有多少 Tmem read、多少本地写、多少 store 未完成，但不能表达当前卡在哪个阶段，也无法精确描述 packet/beat 的可见性。")
    normal("当前模型把 MemUop 收敛成 staged 状态机，至少维护：next_tmem_packet_idx、next_local_beat_idx、各类 staged packet buffer、必要的 valid mask，以及当前所处的阶段。")
    normal("典型阶段包括：FetchFromTmem、WriteLocal、ReadLocalForStore、WriteBackToTmem。")
    normal("这样一来，MMA_LOAD.A/B/C enqueue 之后，排队的是一个显式 target 的宏 load 事务，而不是 compute 微指令；真正逐拍推进的是该事务内部的 packet/beat 状态。")
    normal("同理，MMA_STORE 也不再先 dump 出完整 C tile 再统一写回，而是每拍从 CMem 读取一个逻辑 packet，再通过 TMEM egress 写回一个 64B packet。")
    yellow("【黄字建议补图：MemUop staged 状态机图。建议画出 FetchFromTmem -> WriteLocal -> ReadLocalForStore -> WriteBackToTmem 四个状态，并标注各状态的输入输出资源。】")

    section("8.5 WMMA issue/retire 收紧到 primitive 精度")
    normal("在旧模型中，WMMA 的 issue 条件依赖粗粒度 a_ready/b_ready/c_ready，retire 则主要依赖 async_id 总计数。这种表达方式对功能可以工作，但对时序不够细。")
    normal("当前版本把 issue 和 retire 都收紧到 primitive 精度。dispatch_compute_uop() 只在当前 primitive 对应的 A line、B line、meta line、C subtile 和 tensorcore front-end 都 ready 时才允许发射。")
    normal("同样地，retire_primitive() 也不再简单把结果记到账外 accumulator，而是明确执行 CMem(fp22) 上的 read-modify-write，并在最后一个相关 primitive 完成后才更新 c_wmma_inflight、cmem_final_valid 和 async_id 完成状态。")
    normal("这种收紧保证了三件事：第一，issue 不会因为粗粒度 ready 而过早；第二，completion 不会因为总计数而过早；第三，MMA_STORE 只会在 C 数据真正稳定后启动。")

    section("8.6 TMEM 的物理映射层、共享接口与 bank 仲裁")
    normal("TMEM 当前被明确拆成三层：allocation/resource 层、逻辑 window 层、物理 bank 层。")
    normal("allocation 层回答“这次分配了多少 logical columns”；window 层回答“当前数学矩阵被映射成哪些 logical lines 与 packets”；bank 层回答“这些 logical bytes 最终落到哪些物理 bank 与 physical row”。")
    normal("当前物理配置为 16 banks x 64 rows x 64b，逻辑视图为 64 columns x 128 lines。每个 64B packet 在 bank mapper 中被拆成 8 个 8B lane，再通过 swizzle 分散到多个 bank。")
    normal("对外接口层继续维持共享 512bit ingress 和共享 512bit egress，各方向先按 1 packet/cycle 限流；在接口准入之后，再检查目标 bank 的 1R1W 端口预算。")
    normal("因此，外部可见吞吐和内部 bank 并行能力被明确分离：内部 bank 可以更均匀地承担物理访问，外部仍由单一 512bit packet 通路约束。")
    yellow("【黄字建议补图：TMEM 三层映射图。建议画出数学矩阵 -> window planner -> logical packet/line -> bank mapper -> 16-bank 物理阵列，并标出 ingress/egress 和 1R1W 端口。】")

    section("8.7 CMem(fp22) 与共享物理本地存储")
    normal("本轮修改的一个关键点是把 C 累加域内建到 CMem 本身。CMem 现在按 fp22 存储与累加，fp16/fp32 到 fp22 的转换前移到写入 CMem 前，fp22 回到 fp16/fp32 的转换则保留在导出和写回路径。")
    normal("AMem、BMem、CMem、MetaMem 的底层物理阵列也统一收敛为共享视图。slot0 对应前半深度，slot1 对应后半深度；这样 slot 逻辑不变，但底层不再需要两套完全独立的对象。")
    normal("这种做法的收益是：一方面更接近真实硬件中共享 SRAM + 地址窗口的组织方式；另一方面也使得本地端口仲裁、valid bit 管理和 beat 写入逻辑更集中，更适合做 cycle-accurate 建模。")

    section("8.8 兼容性与实现边界")
    normal("本轮修改始终坚持兼容当前 simx 的时序推进方式，不重起炉灶。新增状态只挂接在 Core::tick / advance_async_tensor_ops 与 TensorUnit::tick / service_mem_ops / dispatch_compute_uop / tick_tensorcore 这几处现有入口。")
    normal("同时，本轮主线收敛以 dense path 为优先。sparse path 的 metadata 搬运、meta shadow window、MetaMem 组织已逐步纳入，但稀疏计算前端仍允许分阶段接回。")
    normal("因此，第 8 章的核心成果不是“把所有特性一次做完”，而是把 packet/beat/primitive 级时序骨架完整搭好，为后续继续收敛 sparse、block scaling 等特性提供稳定基础。")

    chapter("9 BUG列表与收敛情况")
    normal("本章保留原有 BUG 列表的记录方式，但进一步补充每项问题的根因、修复方式和当前状态，避免 bug 记录与语义契约混淆。")
    normal("需要特别强调的是：BUG 列表用于记录实现偏差和测试问题，不等于架构语义本身有误。某些问题的根因来自测试配置、非法参数或旧接口与新模型之间的不匹配。")
    normal("当前已确认并收敛的一项典型问题如下。")
    normal("问题：tcu_mbarrier_async 测试超时。")
    normal("现象：测试在 mma_load 之后没有正确等待异步完成，后续 wmma 提前发射，在 C 尚未 ready 时不断 retry，最终表现为超时。")
    normal("根因：根因不是 TMA 完成语义本身，也不是 TC_COMMIT 与 mbarrier 的设计逻辑出错，而是测试使用了越界的 barrier id。当前 simx 的 barrier 数量由 NUM_BARRIERS = UP(NUM_WARPS/2) 决定，在 4 warp 配置下只有 2 个 barrier，只支持 0 和 1。原测试却使用了 0、2、3、4、1。")
    normal("定位过程：增加日志后可以确认，mbarrier_init(2,1) 与 mbarrier_init(3,1) 都属于 invalid init；后续 tc_commit(2) 与 mbarrier_wait(2) 也都落到了无效 barrier 上，因此并没有真正建立等待关系。")
    normal("修复方式：将测试改为复用单个有效 barrier，使整个等待链条与当前硬件配置保持一致。")
    normal("修复后结果：该测试已经在 simx 下通过，说明 TMA + TC_COMMIT + MBAR_ARRIVE + MBAR_WAIT 这条语义主链是成立的。")
    normal("这一案例的意义在于说明：在当前这种 heavily parameterized 的 Cmodel 中，文档、测试和配置必须同步演进。某些表面看似“硬件功能异常”的问题，实际来自测试越界或旧假设未同步更新。")
    normal("因此，后续 bug 记录建议统一采用“问题现象 / 根因 / 修复方式 / 当前状态 / 是否影响架构语义”这一格式。这样既能保留问题追踪价值，也不会把临时性问题直接写成长期语义。")

    return blocks


def replace_blocks(docx_path: Path, output_path: Path | None = None) -> None:
    with ZipFile(docx_path, "r") as zin:
        xml = zin.read("word/document.xml")
        others = {name: zin.read(name) for name in zin.namelist() if name != "word/document.xml"}

    root = ET.fromstring(xml)
    body = root.find("w:body", NS)
    assert body is not None
    children = list(body)

    start = 1053
    end = 2451
    new_blocks = build_blocks(children)

    for idx in range(end - 1, start - 1, -1):
        body.remove(children[idx])

    insert_pos = start
    for block in new_blocks:
        body.insert(insert_pos, block)
        insert_pos += 1

    new_xml = ET.tostring(root, encoding="utf-8", xml_declaration=True)

    final_output = output_path or docx_path

    with TemporaryDirectory() as td:
        out = Path(td) / final_output.name
        with ZipFile(out, "w") as zout:
            for name, data in others.items():
                zout.writestr(name, data)
            zout.writestr("word/document.xml", new_xml)
        final_output.write_bytes(out.read_bytes())


if __name__ == "__main__":
    replace_blocks(DOCX_PATH, DEFAULT_OUTPUT_PATH)
