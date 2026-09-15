library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library unisim;
use unisim.vcomponents.all;

entity microplastic_patch_qnn_spatial_uart_top is
    port (
        CLK12MHZ     : in  std_logic;
        uart_rxd_out : out std_logic;
        uart_txd_in  : in  std_logic;
        led          : out std_logic_vector(3 downto 0)
    );
end entity microplastic_patch_qnn_spatial_uart_top;

architecture rtl of microplastic_patch_qnn_spatial_uart_top is
    component finn_design_wrapper is
        port (
            ap_clk          : in  std_logic;
            ap_rst_n        : in  std_logic;
            m_axis_0_tdata  : out std_logic_vector(7 downto 0);
            m_axis_0_tready : in  std_logic;
            m_axis_0_tvalid : out std_logic;
            s_axis_0_tdata  : in  std_logic_vector(7 downto 0);
            s_axis_0_tready : out std_logic;
            s_axis_0_tvalid : in  std_logic
        );
    end component;

    signal clk_feedback        : std_logic;
    signal clk_feedback_buf    : std_logic;
    signal clk100_unbuffered   : std_logic;
    signal clk100              : std_logic;
    signal mmcm_locked         : std_logic;
    signal reset_pipe          : std_logic_vector(3 downto 0) := (others => '0');
    signal system_reset_n      : std_logic;
    signal bridge_core_reset_n : std_logic;
    signal core_reset_n        : std_logic;

    signal input_data  : std_logic_vector(7 downto 0);
    signal input_valid : std_logic;
    signal input_ready : std_logic;

    signal raw_output_data  : std_logic_vector(7 downto 0);
    signal raw_output_valid : std_logic;
    signal raw_output_ready : std_logic;
    signal uart_output_data : std_logic_vector(31 downto 0);
begin
    mmcm_inst : MMCME2_BASE
        generic map (
            BANDWIDTH        => "OPTIMIZED",
            CLKFBOUT_MULT_F  => 62.5,
            CLKIN1_PERIOD    => 83.333,
            CLKOUT0_DIVIDE_F => 7.5,
            DIVCLK_DIVIDE    => 1,
            STARTUP_WAIT     => false
        )
        port map (
            CLKIN1   => CLK12MHZ,
            CLKFBIN  => clk_feedback_buf,
            RST      => '0',
            PWRDWN   => '0',
            CLKFBOUT => clk_feedback,
            CLKOUT0  => clk100_unbuffered,
            LOCKED   => mmcm_locked
        );

    feedback_bufg : BUFG
        port map (I => clk_feedback, O => clk_feedback_buf);

    output_bufg : BUFG
        port map (I => clk100_unbuffered, O => clk100);

    process (clk100, mmcm_locked)
    begin
        if mmcm_locked = '0' then
            reset_pipe <= (others => '0');
        elsif rising_edge(clk100) then
            reset_pipe <= reset_pipe(2 downto 0) & '1';
        end if;
    end process;
    system_reset_n <= reset_pipe(3);
    core_reset_n <= system_reset_n and bridge_core_reset_n;

    uart_output_data <= std_logic_vector(resize(signed(raw_output_data), 32));

    bridge_inst : entity work.finn_uart_bridge
        generic map (
            CLKS_PER_BIT => 100,
            INPUT_BYTES  => 1024,
            OUTPUT_WORDS => 1,
            OUTPUT_BITS  => 32
        )
        port map (
            clk           => clk100,
            reset_n       => system_reset_n,
            uart_rx_i     => uart_txd_in,
            uart_tx_o     => uart_rxd_out,
            core_reset_n  => bridge_core_reset_n,
            s_axis_tdata  => input_data,
            s_axis_tvalid => input_valid,
            s_axis_tready => input_ready,
            m_axis_tdata  => uart_output_data,
            m_axis_tvalid => raw_output_valid,
            m_axis_tready => raw_output_ready,
            status_led    => led
        );

    classifier_inst : finn_design_wrapper
        port map (
            ap_clk          => clk100,
            ap_rst_n        => core_reset_n,
            s_axis_0_tdata  => input_data,
            s_axis_0_tvalid => input_valid,
            s_axis_0_tready => input_ready,
            m_axis_0_tdata  => raw_output_data,
            m_axis_0_tvalid => raw_output_valid,
            m_axis_0_tready => raw_output_ready
        );
end architecture rtl;
