library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity hybrid_dual_roi_input_router is
    generic (
        ROI_BYTES : positive := 9216
    );
    port (
        clk     : in std_logic;
        reset_n : in std_logic;

        s_axis_tdata  : in  std_logic_vector(7 downto 0);
        s_axis_tvalid : in  std_logic;
        s_axis_tready : out std_logic;

        qnn_tdata  : out std_logic_vector(7 downto 0);
        qnn_tvalid : out std_logic;
        qnn_tready : in  std_logic;

        classical_tdata  : out std_logic_vector(7 downto 0);
        classical_tvalid : out std_logic;
        classical_tready : in  std_logic
    );
end entity hybrid_dual_roi_input_router;

architecture rtl of hybrid_dual_roi_input_router is
    constant TOTAL_BYTES : positive := 2 * ROI_BYTES;
    signal byte_index : natural range 0 to TOTAL_BYTES := 0;
    signal selected_ready : std_logic;
begin
    qnn_tdata <= s_axis_tdata;
    classical_tdata <= s_axis_tdata;

    qnn_tvalid <= s_axis_tvalid
        when byte_index < ROI_BYTES else '0';
    classical_tvalid <= s_axis_tvalid
        when byte_index >= ROI_BYTES and byte_index < TOTAL_BYTES else '0';
    selected_ready <= qnn_tready
        when byte_index < ROI_BYTES else
        classical_tready when byte_index < TOTAL_BYTES else
        '0';
    s_axis_tready <= selected_ready;

    process (clk)
    begin
        if rising_edge(clk) then
            if reset_n = '0' then
                byte_index <= 0;
            elsif s_axis_tvalid = '1' and selected_ready = '1' then
                if byte_index < TOTAL_BYTES then
                    byte_index <= byte_index + 1;
                end if;
            end if;
        end if;
    end process;
end architecture rtl;
