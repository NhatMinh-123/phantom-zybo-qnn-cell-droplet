// Marks each fixed-size FINN raw-head result as one AXI-Stream packet.
// The QNN produces 24-bit words and has no TLAST.  For the 96x96 model,
// 15 * 24 * 24 values are packed into 2,880 output transfers.
module axis_tlast_injector_24 #(
    parameter integer WORDS_PER_FRAME = 8640
) (
    input  wire        aclk,
    input  wire        aresetn,
    input  wire [23:0] s_axis_tdata,
    input  wire        s_axis_tvalid,
    output wire        s_axis_tready,
    output wire [23:0] m_axis_tdata,
    output wire [2:0]  m_axis_tkeep,
    output wire        m_axis_tvalid,
    input  wire        m_axis_tready,
    output wire        m_axis_tlast
);
    localparam integer COUNT_WIDTH = $clog2(WORDS_PER_FRAME);
    reg [COUNT_WIDTH-1:0] word_count;

    assign s_axis_tready = m_axis_tready;
    assign m_axis_tdata  = s_axis_tdata;
    assign m_axis_tkeep  = 3'b111;
    assign m_axis_tvalid = s_axis_tvalid;
    assign m_axis_tlast  = (word_count == WORDS_PER_FRAME - 1);

    always @(posedge aclk) begin
        if (!aresetn) begin
            word_count <= {COUNT_WIDTH{1'b0}};
        end else if (s_axis_tvalid && m_axis_tready) begin
            if (word_count == WORDS_PER_FRAME - 1)
                word_count <= {COUNT_WIDTH{1'b0}};
            else
                word_count <= word_count + 1'b1;
        end
    end
endmodule
