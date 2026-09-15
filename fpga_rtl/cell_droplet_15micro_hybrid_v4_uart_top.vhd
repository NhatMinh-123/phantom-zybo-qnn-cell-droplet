library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library unisim;
use unisim.vcomponents.all;

entity cell_droplet_15micro_hybrid_v4_uart_top is
    port (
        CLK12MHZ     : in std_logic;
        uart_rxd_out : out std_logic;
        uart_txd_in  : in std_logic;
        led          : out std_logic_vector(3 downto 0)
    );
end entity cell_droplet_15micro_hybrid_v4_uart_top;

architecture rtl of cell_droplet_15micro_hybrid_v4_uart_top is
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

    function clipped_byte(value : integer) return std_logic_vector is
        variable clipped : integer;
    begin
        clipped := value;
        if clipped < 0 then
            clipped := 0;
        elsif clipped > 95 then
            clipped := 95;
        end if;
        return std_logic_vector(to_unsigned(clipped, 8));
    end function;

    function make_classical_box_record(
        class_id : natural;
        score : std_logic_vector(7 downto 0);
        center_x : integer;
        center_y : integer;
        half_size : integer
    ) return std_logic_vector is
        variable result : std_logic_vector(63 downto 0) := (others => '0');
    begin
        result(7 downto 0) := x"43"; -- C
        result(15 downto 8) := x"42"; -- B
        result(23 downto 16) := std_logic_vector(to_unsigned(class_id, 8));
        result(31 downto 24) := score;
        result(39 downto 32) := clipped_byte(center_x - half_size);
        result(47 downto 40) := clipped_byte(center_y - half_size);
        result(55 downto 48) := clipped_byte(center_x + half_size);
        result(63 downto 56) := clipped_byte(center_y + half_size);
        return result;
    end function;

    signal clk_feedback : std_logic;
    signal clk_feedback_buf : std_logic;
    signal clk96_unbuffered : std_logic;
    signal clk96 : std_logic;
    signal mmcm_locked : std_logic;
    signal reset_pipe : std_logic_vector(3 downto 0) := (others => '0');
    signal system_reset_n : std_logic;
    signal bridge_core_reset_n : std_logic;
    signal core_reset_n : std_logic;
    signal previous_bridge_reset_n : std_logic := '0';
    signal frame_start : std_logic := '0';

    signal bridge_input_data : std_logic_vector(7 downto 0);
    signal bridge_input_valid : std_logic;
    signal bridge_input_ready : std_logic;
    signal qnn_input_data : std_logic_vector(7 downto 0);
    signal qnn_input_valid : std_logic;
    signal qnn_input_ready : std_logic;
    signal classical_input_data : std_logic_vector(7 downto 0);
    signal classical_input_valid : std_logic;
    signal classical_input_ready : std_logic;

    signal raw_output_data : std_logic_vector(23 downto 0);
    signal raw_output_valid : std_logic;
    signal raw_output_ready : std_logic;
    signal requant_data : std_logic_vector(7 downto 0);
    signal requant_valid : std_logic;
    signal requant_ready : std_logic;
    signal guarded_data : std_logic_vector(7 downto 0);
    signal guarded_valid : std_logic;
    signal guarded_ready : std_logic;

    signal qnn_frame_done : std_logic;
    signal qnn_droplet_present : std_logic;
    signal qnn_droplet_y : std_logic_vector(6 downto 0);
    signal qnn_droplet_score : std_logic_vector(7 downto 0);
    signal qnn_cell_count : std_logic_vector(1 downto 0);
    signal qnn_cell0_y : std_logic_vector(6 downto 0);
    signal qnn_cell0_score : std_logic_vector(7 downto 0);
    signal qnn_cell1_y : std_logic_vector(6 downto 0);
    signal qnn_cell1_score : std_logic_vector(7 downto 0);

    signal classical_frame_done : std_logic;
    signal classical_droplet_score : std_logic_vector(15 downto 0);
    signal classical_droplet_y : std_logic_vector(6 downto 0);
    signal classical_droplet_radius : std_logic_vector(6 downto 0);
    signal classical_droplet_present : std_logic;
    signal classical_cell_count : std_logic_vector(1 downto 0);
    signal classical_cell0_x : std_logic_vector(6 downto 0);
    signal classical_cell0_y : std_logic_vector(6 downto 0);
    signal classical_cell0_score : std_logic_vector(7 downto 0);
    signal classical_cell1_x : std_logic_vector(6 downto 0);
    signal classical_cell1_y : std_logic_vector(6 downto 0);
    signal classical_cell1_score : std_logic_vector(7 downto 0);
    signal classical_box_valid : std_logic_vector(2 downto 0);
    signal classical_box0_record : std_logic_vector(63 downto 0);
    signal classical_box1_record : std_logic_vector(63 downto 0);
    signal classical_box2_record : std_logic_vector(63 downto 0);

    signal hybrid_status_valid : std_logic;
    signal hybrid_status_flags : std_logic_vector(7 downto 0);
    signal hybrid_droplet_count : std_logic_vector(15 downto 0);
    signal hybrid_cell_count : std_logic_vector(15 downto 0);
begin
    process (all)
        variable droplet_score_code : natural range 0 to 8191;
        variable cell_count_value : natural range 0 to 3;
    begin
        droplet_score_code :=
            to_integer(unsigned(classical_droplet_score)) / 8;
        if droplet_score_code > 255 then
            droplet_score_code := 255;
        end if;
        cell_count_value := to_integer(unsigned(classical_cell_count));

        classical_box_valid(0) <= classical_droplet_present;
        if cell_count_value >= 1 then
            classical_box_valid(1) <= '1';
        else
            classical_box_valid(1) <= '0';
        end if;
        if cell_count_value >= 2 then
            classical_box_valid(2) <= '1';
        else
            classical_box_valid(2) <= '0';
        end if;
        classical_box0_record <= make_classical_box_record(
            1,
            std_logic_vector(to_unsigned(droplet_score_code, 8)),
            48,
            to_integer(unsigned(classical_droplet_y)),
            to_integer(unsigned(classical_droplet_radius)) + 4
        );
        classical_box1_record <= make_classical_box_record(
            0,
            classical_cell0_score,
            to_integer(unsigned(classical_cell0_x)),
            to_integer(unsigned(classical_cell0_y)),
            5
        );
        classical_box2_record <= make_classical_box_record(
            0,
            classical_cell1_score,
            to_integer(unsigned(classical_cell1_x)),
            to_integer(unsigned(classical_cell1_y)),
            5
        );
    end process;

    mmcm_inst : MMCME2_BASE
        generic map (
            BANDWIDTH => "OPTIMIZED",
            CLKFBOUT_MULT_F => 50.0,
            CLKIN1_PERIOD => 83.333,
            CLKOUT0_DIVIDE_F => 6.25,
            DIVCLK_DIVIDE => 1,
            STARTUP_WAIT => false
        )
        port map (
            CLKIN1 => CLK12MHZ,
            CLKFBIN => clk_feedback_buf,
            RST => '0',
            PWRDWN => '0',
            CLKFBOUT => clk_feedback,
            CLKOUT0 => clk96_unbuffered,
            LOCKED => mmcm_locked
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
            else
                frame_start <= bridge_core_reset_n and
                    not previous_bridge_reset_n;
                previous_bridge_reset_n <= bridge_core_reset_n;
            end if;
        end if;
    end process;

    bridge_inst : entity work.finn_uart_sparse_stream_bridge
        generic map (
            CLKS_PER_BIT => 8,
            INPUT_BYTES => 18432,
            OUTPUT_WORDS => 8640,
            OUTPUT_CHANNELS => 15,
            GRID_POINTS => 576,
            INPUT_FIFO_DEPTH => 4096,
            MAX_CANDIDATES => 256,
            CELL_OBJECT_CODE => 105,
            DROPLET_OBJECT_CODE => 66,
            HYBRID_STATUS_ENABLE => true
        )
        port map (
            clk => clk96,
            reset_n => system_reset_n,
            uart_rx_i => uart_txd_in,
            uart_tx_o => uart_rxd_out,
            core_reset_n => bridge_core_reset_n,
            s_axis_tdata => bridge_input_data,
            s_axis_tvalid => bridge_input_valid,
            s_axis_tready => bridge_input_ready,
            m_axis_tdata => guarded_data,
            m_axis_tvalid => guarded_valid,
            m_axis_tready => guarded_ready,
            hybrid_status_valid => hybrid_status_valid,
            hybrid_status_flags => hybrid_status_flags,
            hybrid_droplet_count => hybrid_droplet_count,
            hybrid_cell_count => hybrid_cell_count,
            hybrid_box_valid => classical_box_valid,
            hybrid_box0_record => classical_box0_record,
            hybrid_box1_record => classical_box1_record,
            hybrid_box2_record => classical_box2_record,
            status_led => led
        );

    input_router_inst : entity work.hybrid_dual_roi_input_router
        generic map (ROI_BYTES => 9216)
        port map (
            clk => clk96,
            reset_n => core_reset_n,
            s_axis_tdata => bridge_input_data,
            s_axis_tvalid => bridge_input_valid,
            s_axis_tready => bridge_input_ready,
            qnn_tdata => qnn_input_data,
            qnn_tvalid => qnn_input_valid,
            qnn_tready => qnn_input_ready,
            classical_tdata => classical_input_data,
            classical_tvalid => classical_input_valid,
            classical_tready => classical_input_ready
        );

    detector_inst : finn_design_wrapper
        port map (
            ap_clk => clk96,
            ap_rst_n => core_reset_n,
            s_axis_0_tdata => qnn_input_data,
            s_axis_0_tvalid => qnn_input_valid,
            s_axis_0_tready => qnn_input_ready,
            m_axis_0_tdata => raw_output_data,
            m_axis_0_tvalid => raw_output_valid,
            m_axis_0_tready => raw_output_ready
        );

    requantizer_inst : entity work.detector_output_requantizer_15micro
        port map (
            clk => clk96,
            reset_n => core_reset_n,
            s_axis_tdata => raw_output_data,
            s_axis_tvalid => raw_output_valid,
            s_axis_tready => raw_output_ready,
            m_axis_tdata => requant_data,
            m_axis_tvalid => requant_valid,
            m_axis_tready => requant_ready
        );

    radial_guard_inst : entity work.qnn_cell_radial_guard_filter_v3
        generic map (
            IMAGE_WIDTH => 96,
            IMAGE_HEIGHT => 96,
            GRID_WIDTH => 24,
            GRID_HEIGHT => 24,
            GRID_STRIDE => 4,
            GRID_CENTER_OFFSET => 2,
            CELL_LOW_OBJECT_CODE => 66,
            CELL_HIGH_OBJECT_CODE => 105,
            RADIAL_THRESHOLD_X8 => 224,
            SUPPORT_RADIUS => 1
        )
        port map (
            clk => clk96,
            reset_n => core_reset_n,
            image_tdata => qnn_input_data,
            image_tvalid => qnn_input_valid,
            image_tready => qnn_input_ready,
            s_axis_tdata => requant_data,
            s_axis_tvalid => requant_valid,
            s_axis_tready => requant_ready,
            m_axis_tdata => guarded_data,
            m_axis_tvalid => guarded_valid,
            m_axis_tready => guarded_ready
        );

    qnn_observer_inst : entity work.qnn_gate_observer
        generic map (
            GATE_COLUMN_MIN => 11,
            GATE_COLUMN_MAX => 14,
            CELL_OBJECT_CODE => 105,
            DROPLET_OBJECT_CODE => 66
        )
        port map (
            clk => clk96,
            reset_n => core_reset_n,
            stream_tdata => guarded_data,
            stream_tvalid => guarded_valid,
            stream_tready => guarded_ready,
            frame_done => qnn_frame_done,
            droplet_present => qnn_droplet_present,
            droplet_y => qnn_droplet_y,
            droplet_score => qnn_droplet_score,
            cell_count => qnn_cell_count,
            cell0_y => qnn_cell0_y,
            cell0_score => qnn_cell0_score,
            cell1_y => qnn_cell1_y,
            cell1_score => qnn_cell1_score
        );

    classical_analyzer_inst : entity work.classical_roi_gate_analyzer
        generic map (
            IMAGE_SIZE => 96,
            GATE_X => 48,
            DROPLET_SCORE_THRESHOLD => 500,
            CELL_RESPONSE_THRESHOLD_X8 => 320,
            INPUT_RGB332 => true
        )
        port map (
            clk => clk96,
            reset_n => core_reset_n,
            s_axis_tdata => classical_input_data,
            s_axis_tvalid => classical_input_valid,
            s_axis_tready => classical_input_ready,
            frame_done => classical_frame_done,
            droplet_score => classical_droplet_score,
            droplet_center_y => classical_droplet_y,
            droplet_radius => classical_droplet_radius,
            droplet_present => classical_droplet_present,
            cell_count => classical_cell_count,
            cell0_x => classical_cell0_x,
            cell0_y => classical_cell0_y,
            cell0_score => classical_cell0_score,
            cell1_x => classical_cell1_x,
            cell1_y => classical_cell1_y,
            cell1_score => classical_cell1_score
        );

    consensus_inst : entity work.hybrid_temporal_consensus_counter
        generic map (
            HISTORY_DEPTH => 64,
            MINIMUM_DELAY_FRAMES => 1,
            MAXIMUM_DELAY_FRAMES => 60,
            NOMINAL_DELAY_FRAMES => 12,
            MAXIMUM_Y_DISTANCE => 29,
            DROPLET_REFRACTORY_FRAMES => 7,
            CELL_REQUIRED_HITS => 2,
            CELL_MAXIMUM_Y_MOTION => 7,
            CELL_REFRACTORY_FRAMES => 7,
            QNN_DROPLET_THRESHOLD => 66,
            CLASSICAL_DROP_THRESHOLD => 500
        )
        port map (
            clk => clk96,
            reset_n => system_reset_n,
            frame_start => frame_start,
            qnn_frame_done => qnn_frame_done,
            qnn_droplet_present => qnn_droplet_present,
            qnn_droplet_y => qnn_droplet_y,
            qnn_droplet_score => qnn_droplet_score,
            qnn_cell_count => qnn_cell_count,
            qnn_cell0_y => qnn_cell0_y,
            qnn_cell1_y => qnn_cell1_y,
            classical_frame_done => classical_frame_done,
            classical_droplet_present => classical_droplet_present,
            classical_droplet_y => classical_droplet_y,
            classical_droplet_score => classical_droplet_score,
            classical_cell_count => classical_cell_count,
            classical_cell0_y => classical_cell0_y,
            classical_cell1_y => classical_cell1_y,
            status_valid => hybrid_status_valid,
            status_flags => hybrid_status_flags,
            droplet_count => hybrid_droplet_count,
            cell_count => hybrid_cell_count
        );
end architecture rtl;
