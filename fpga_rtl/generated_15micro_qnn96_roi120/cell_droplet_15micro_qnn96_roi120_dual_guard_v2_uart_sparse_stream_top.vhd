library ieee;
use ieee.std_logic_1164.all;

library unisim;
use unisim.vcomponents.all;

entity cell_droplet_15micro_qnn96_roi120_dual_guard_v2_uart_sparse_stream_top is
    port (
        CLK12MHZ     : in  std_logic;
        uart_rxd_out : out std_logic;
        uart_txd_in  : in  std_logic;
        led          : out std_logic_vector(3 downto 0)
    );
end entity cell_droplet_15micro_qnn96_roi120_dual_guard_v2_uart_sparse_stream_top;

architecture rtl of
    cell_droplet_15micro_qnn96_roi120_dual_guard_v2_uart_sparse_stream_top is
    component finn_design_wrapper is
        port (
            ap_clk          : in  std_logic;
            ap_rst_n        : in  std_logic;
            m_axis_0_tdata  : out std_logic_vector(23 downto 0);
            m_axis_0_tready : in  std_logic;
            m_axis_0_tvalid : out std_logic;
            s_axis_0_tdata  : in  std_logic_vector(7 downto 0);
            s_axis_0_tready : out std_logic;
            s_axis_0_tvalid : in  std_logic
        );
    end component;

    signal clk_feedback        : std_logic;
    signal clk_feedback_buf    : std_logic;
    signal clk108_unbuffered   : std_logic;
    signal clk108              : std_logic;
    signal mmcm_locked         : std_logic;
    signal reset_pipe          : std_logic_vector(3 downto 0) := (others => '0');
    signal system_reset_n      : std_logic;
    signal bridge_core_reset_n : std_logic;
    signal core_reset_n        : std_logic;
    signal input_data          : std_logic_vector(7 downto 0);
    signal input_valid         : std_logic;
    signal input_ready         : std_logic;
    signal raw_output_data     : std_logic_vector(23 downto 0);
    signal raw_output_valid    : std_logic;
    signal raw_output_ready    : std_logic;
    signal requant_data        : std_logic_vector(7 downto 0);
    signal requant_valid       : std_logic;
    signal requant_ready       : std_logic;
    signal guarded_data        : std_logic_vector(7 downto 0);
    signal guarded_valid       : std_logic;
    signal guarded_ready       : std_logic;
begin
    mmcm_inst : MMCME2_BASE
        generic map (
            BANDWIDTH        => "OPTIMIZED",
            CLKFBOUT_MULT_F  => 54.0,
            CLKIN1_PERIOD    => 83.333,
            CLKOUT0_DIVIDE_F => 6.0,
            DIVCLK_DIVIDE    => 1,
            STARTUP_WAIT     => false
        )
        port map (
            CLKIN1   => CLK12MHZ,
            CLKFBIN  => clk_feedback_buf,
            RST      => '0',
            PWRDWN   => '0',
            CLKFBOUT => clk_feedback,
            CLKOUT0  => clk108_unbuffered,
            LOCKED   => mmcm_locked
        );

    feedback_bufg : BUFG port map (I => clk_feedback, O => clk_feedback_buf);
    output_bufg : BUFG port map (I => clk108_unbuffered, O => clk108);

    process (clk108, mmcm_locked)
    begin
        if mmcm_locked = '0' then
            reset_pipe <= (others => '0');
        elsif rising_edge(clk108) then
            reset_pipe <= reset_pipe(2 downto 0) & '1';
        end if;
    end process;
    system_reset_n <= reset_pipe(3);
    core_reset_n <= system_reset_n and bridge_core_reset_n;

    bridge_inst : entity work.finn_uart_sparse_stream_bridge
        generic map (
            CLKS_PER_BIT        => 9,
            INPUT_BYTES         => 9216,
            OUTPUT_WORDS        => 8640,
            OUTPUT_CHANNELS     => 15,
            GRID_POINTS         => 576,
            INPUT_FIFO_DEPTH    => 4096,
            MAX_CANDIDATES      => 256,
            CELL_OBJECT_CODE    => 105,
            DROPLET_OBJECT_CODE => 66
        )
        port map (
            clk => clk108,
            reset_n => system_reset_n,
            uart_rx_i => uart_txd_in,
            uart_tx_o => uart_rxd_out,
            core_reset_n => bridge_core_reset_n,
            s_axis_tdata => input_data,
            s_axis_tvalid => input_valid,
            s_axis_tready => input_ready,
            m_axis_tdata => guarded_data,
            m_axis_tvalid => guarded_valid,
            m_axis_tready => guarded_ready,
            status_led => led
        );

    requantizer_inst : entity work.detector_output_requantizer_15micro
        port map (
            clk => clk108,
            reset_n => core_reset_n,
            s_axis_tdata => raw_output_data,
            s_axis_tvalid => raw_output_valid,
            s_axis_tready => raw_output_ready,
            m_axis_tdata => requant_data,
            m_axis_tvalid => requant_valid,
            m_axis_tready => requant_ready
        );

    radial_guard_inst : entity work.qnn_cell_radial_guard_filter_v2
        generic map (
            IMAGE_WIDTH           => 96,
            IMAGE_HEIGHT          => 96,
            GRID_WIDTH            => 24,
            GRID_HEIGHT           => 24,
            GRID_STRIDE           => 4,
            GRID_CENTER_OFFSET    => 2,
            CELL_LOW_OBJECT_CODE  => 66,
            CELL_HIGH_OBJECT_CODE => 105,
            RADIAL_THRESHOLD_X8   => 224,
            SUPPORT_RADIUS        => 1
        )
        port map (
            clk => clk108,
            reset_n => core_reset_n,
            image_tdata => input_data,
            image_tvalid => input_valid,
            image_tready => input_ready,
            s_axis_tdata => requant_data,
            s_axis_tvalid => requant_valid,
            s_axis_tready => requant_ready,
            m_axis_tdata => guarded_data,
            m_axis_tvalid => guarded_valid,
            m_axis_tready => guarded_ready
        );

    detector_inst : finn_design_wrapper
        port map (
            ap_clk => clk108,
            ap_rst_n => core_reset_n,
            s_axis_0_tdata => input_data,
            s_axis_0_tvalid => input_valid,
            s_axis_0_tready => input_ready,
            m_axis_0_tdata => raw_output_data,
            m_axis_0_tvalid => raw_output_valid,
            m_axis_0_tready => raw_output_ready
        );
end architecture rtl;
