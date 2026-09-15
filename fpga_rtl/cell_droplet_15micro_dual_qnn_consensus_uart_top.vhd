library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library unisim;
use unisim.vcomponents.all;

-- Two adjacent spatial decisions reuse one QNN core. UART requests alternate
-- ROI A and ROI B; temporal consensus and cumulative counters remain on FPGA.
entity cell_droplet_15micro_dual_qnn_consensus_uart_top is
    port (
        CLK12MHZ     : in std_logic;
        uart_rxd_out : out std_logic;
        uart_txd_in  : in std_logic;
        led          : out std_logic_vector(3 downto 0)
    );
end entity cell_droplet_15micro_dual_qnn_consensus_uart_top;

architecture rtl of cell_droplet_15micro_dual_qnn_consensus_uart_top is
    component finn_design_wrapper is
        port (
            ap_clk          : in std_logic;
            ap_rst_n        : in std_logic;
            m_axis_0_tdata  : out std_logic_vector(23 downto 0);
            m_axis_0_tready : in std_logic;
            m_axis_0_tvalid : out std_logic;
            s_axis_0_tdata  : in std_logic_vector(7 downto 0);
            s_axis_0_tready : out std_logic;
            s_axis_0_tvalid : in std_logic
        );
    end component;

    signal clk_feedback, clk_feedback_buf : std_logic;
    signal clk96_unbuffered, clk96, mmcm_locked : std_logic;
    signal reset_pipe : std_logic_vector(3 downto 0) := (others => '0');
    signal system_reset_n, bridge_core_reset_n, core_reset_n : std_logic;
    signal previous_bridge_reset_n : std_logic := '0';
    signal frame_start : std_logic := '0';
    signal next_roi : std_logic := '0';
    signal active_roi : std_logic := '0';
    signal pair_frame_start : std_logic;

    signal input_data : std_logic_vector(7 downto 0);
    signal input_valid, input_ready : std_logic;
    signal raw_output_data : std_logic_vector(23 downto 0);
    signal raw_output_valid, raw_output_ready : std_logic;
    signal requant_data : std_logic_vector(7 downto 0);
    signal requant_valid, requant_ready : std_logic;
    signal guarded_data : std_logic_vector(7 downto 0);
    signal guarded_valid, guarded_ready : std_logic;

    signal qnn_frame_done, qnn_droplet_present : std_logic;
    signal qnn_droplet_y : std_logic_vector(6 downto 0);
    signal qnn_droplet_score : std_logic_vector(7 downto 0);
    signal qnn_cell_count : std_logic_vector(1 downto 0);
    signal qnn_cell0_y, qnn_cell1_y : std_logic_vector(6 downto 0);
    signal qnn_cell0_score, qnn_cell1_score : std_logic_vector(7 downto 0);
    signal roi_a_done, roi_b_done : std_logic;
    signal roi_b_droplet_score_x8 : std_logic_vector(15 downto 0);

    signal consensus_status_valid : std_logic;
    signal consensus_status_flags : std_logic_vector(7 downto 0);
    signal consensus_droplet_count, consensus_cell_count : std_logic_vector(15 downto 0);
    signal bridge_status_valid : std_logic;
    signal bridge_status_flags : std_logic_vector(7 downto 0);
begin
    mmcm_inst : MMCME2_BASE
        generic map (
            BANDWIDTH => "OPTIMIZED", CLKFBOUT_MULT_F => 50.0,
            CLKIN1_PERIOD => 83.333, CLKOUT0_DIVIDE_F => 6.25,
            DIVCLK_DIVIDE => 1, STARTUP_WAIT => false
        )
        port map (
            CLKIN1 => CLK12MHZ, CLKFBIN => clk_feedback_buf,
            RST => '0', PWRDWN => '0', CLKFBOUT => clk_feedback,
            CLKOUT0 => clk96_unbuffered, LOCKED => mmcm_locked
        );
    feedback_bufg : BUFG port map (I => clk_feedback, O => clk_feedback_buf);
    output_bufg : BUFG port map (I => clk96_unbuffered, O => clk96);

    process (clk96, mmcm_locked)
    begin
        if mmcm_locked = '0' then
            reset_pipe <= (others => '0');
        elsif rising_edge(clk96) then
            reset_pipe <= reset_pipe(2 downto 0) & '1';
        end if;
    end process;
    system_reset_n <= reset_pipe(3);
    core_reset_n <= system_reset_n and bridge_core_reset_n;

    process (clk96)
    begin
        if rising_edge(clk96) then
            if system_reset_n = '0' then
                previous_bridge_reset_n <= '0';
                frame_start <= '0';
                next_roi <= '0';
                active_roi <= '0';
            else
                frame_start <= bridge_core_reset_n and not previous_bridge_reset_n;
                previous_bridge_reset_n <= bridge_core_reset_n;
                if bridge_core_reset_n = '1' and previous_bridge_reset_n = '0' then
                    active_roi <= next_roi;
                    next_roi <= not next_roi;
                end if;
            end if;
        end if;
    end process;

    pair_frame_start <= frame_start when active_roi = '0' else '0';
    roi_a_done <= qnn_frame_done when active_roi = '0' else '0';
    roi_b_done <= qnn_frame_done when active_roi = '1' else '0';
    roi_b_droplet_score_x8 <= "00000" & qnn_droplet_score & "000";

    -- ROI A returns its sparse boxes immediately; ROI B also returns the
    -- consensus decision and cumulative FPGA counters.
    bridge_status_valid <= qnn_frame_done when active_roi = '0'
                           else consensus_status_valid;
    bridge_status_flags <= ('1' & '0' & "0000" &
                            (qnn_cell_count(1) or qnn_cell_count(0)) &
                            qnn_droplet_present) when active_roi = '0'
                           else consensus_status_flags;

    bridge_inst : entity work.finn_uart_sparse_stream_bridge
        generic map (
            CLKS_PER_BIT => 8, INPUT_BYTES => 9216,
            OUTPUT_WORDS => 8640, OUTPUT_CHANNELS => 15,
            GRID_POINTS => 576, INPUT_FIFO_DEPTH => 4096,
            MAX_CANDIDATES => 256, CELL_OBJECT_CODE => 105,
            DROPLET_OBJECT_CODE => 66, HYBRID_STATUS_ENABLE => true
        )
        port map (
            clk => clk96, reset_n => system_reset_n,
            uart_rx_i => uart_txd_in, uart_tx_o => uart_rxd_out,
            core_reset_n => bridge_core_reset_n,
            s_axis_tdata => input_data, s_axis_tvalid => input_valid,
            s_axis_tready => input_ready,
            m_axis_tdata => guarded_data, m_axis_tvalid => guarded_valid,
            m_axis_tready => guarded_ready,
            hybrid_status_valid => bridge_status_valid,
            hybrid_status_flags => bridge_status_flags,
            hybrid_droplet_count => consensus_droplet_count,
            hybrid_cell_count => consensus_cell_count,
            status_led => led
        );

    detector_inst : finn_design_wrapper
        port map (
            ap_clk => clk96, ap_rst_n => core_reset_n,
            s_axis_0_tdata => input_data, s_axis_0_tvalid => input_valid,
            s_axis_0_tready => input_ready,
            m_axis_0_tdata => raw_output_data,
            m_axis_0_tvalid => raw_output_valid,
            m_axis_0_tready => raw_output_ready
        );

    requantizer_inst : entity work.detector_output_requantizer_15micro
        port map (
            clk => clk96, reset_n => core_reset_n,
            s_axis_tdata => raw_output_data,
            s_axis_tvalid => raw_output_valid,
            s_axis_tready => raw_output_ready,
            m_axis_tdata => requant_data,
            m_axis_tvalid => requant_valid,
            m_axis_tready => requant_ready
        );

    radial_guard_inst : entity work.qnn_cell_radial_guard_filter_v3
        generic map (
            IMAGE_WIDTH => 96, IMAGE_HEIGHT => 96,
            GRID_WIDTH => 24, GRID_HEIGHT => 24,
            GRID_STRIDE => 4, GRID_CENTER_OFFSET => 2,
            CELL_LOW_OBJECT_CODE => 66, CELL_HIGH_OBJECT_CODE => 105,
            RADIAL_THRESHOLD_X8 => 224, SUPPORT_RADIUS => 1
        )
        port map (
            clk => clk96, reset_n => core_reset_n,
            image_tdata => input_data, image_tvalid => input_valid,
            image_tready => input_ready,
            s_axis_tdata => requant_data, s_axis_tvalid => requant_valid,
            s_axis_tready => requant_ready,
            m_axis_tdata => guarded_data, m_axis_tvalid => guarded_valid,
            m_axis_tready => guarded_ready
        );

    observer_inst : entity work.qnn_gate_observer
        generic map (
            GATE_COLUMN_MIN => 11, GATE_COLUMN_MAX => 14,
            CELL_OBJECT_CODE => 105, DROPLET_OBJECT_CODE => 66
        )
        port map (
            clk => clk96, reset_n => core_reset_n,
            stream_tdata => guarded_data,
            stream_tvalid => guarded_valid,
            stream_tready => guarded_ready,
            frame_done => qnn_frame_done,
            droplet_present => qnn_droplet_present,
            droplet_y => qnn_droplet_y,
            droplet_score => qnn_droplet_score,
            cell_count => qnn_cell_count,
            cell0_y => qnn_cell0_y, cell0_score => qnn_cell0_score,
            cell1_y => qnn_cell1_y, cell1_score => qnn_cell1_score
        );

    consensus_inst : entity work.hybrid_temporal_consensus_counter
        generic map (
            HISTORY_DEPTH => 64, MINIMUM_DELAY_FRAMES => 1,
            MAXIMUM_DELAY_FRAMES => 60, NOMINAL_DELAY_FRAMES => 12,
            MAXIMUM_Y_DISTANCE => 29, DROPLET_REFRACTORY_FRAMES => 7,
            CELL_REQUIRED_HITS => 2, CELL_MAXIMUM_Y_MOTION => 7,
            CELL_REFRACTORY_FRAMES => 7, QNN_DROPLET_THRESHOLD => 66,
            CLASSICAL_DROP_THRESHOLD => 528
        )
        port map (
            clk => clk96, reset_n => system_reset_n,
            frame_start => pair_frame_start,
            qnn_frame_done => roi_a_done,
            qnn_droplet_present => qnn_droplet_present,
            qnn_droplet_y => qnn_droplet_y,
            qnn_droplet_score => qnn_droplet_score,
            qnn_cell_count => qnn_cell_count,
            qnn_cell0_y => qnn_cell0_y, qnn_cell1_y => qnn_cell1_y,
            classical_frame_done => roi_b_done,
            classical_droplet_present => qnn_droplet_present,
            classical_droplet_y => qnn_droplet_y,
            classical_droplet_score => roi_b_droplet_score_x8,
            classical_cell_count => qnn_cell_count,
            classical_cell0_y => qnn_cell0_y,
            classical_cell1_y => qnn_cell1_y,
            status_valid => consensus_status_valid,
            status_flags => consensus_status_flags,
            droplet_count => consensus_droplet_count,
            cell_count => consensus_cell_count
        );
end architecture rtl;
