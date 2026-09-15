library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library unisim;
use unisim.vcomponents.all;

entity bright_round_feature_uart_top is
    port (
        CLK12MHZ     : in  std_logic;
        uart_rxd_out : out std_logic;
        uart_txd_in  : in  std_logic;
        led          : out std_logic_vector(3 downto 0)
    );
end entity bright_round_feature_uart_top;

architecture rtl of bright_round_feature_uart_top is
    constant REQUEST_MAGIC  : std_logic_vector(7 downto 0) := x"A5";
    constant RESPONSE_MAGIC : std_logic_vector(7 downto 0) := x"5A";

    signal clk_feedback      : std_logic;
    signal clk_feedback_buf  : std_logic;
    signal clk100_unbuffered : std_logic;
    signal clk100            : std_logic;
    signal mmcm_locked       : std_logic;
    signal reset_pipe        : std_logic_vector(3 downto 0) := (others => '0');
    signal system_reset_n    : std_logic;

    signal rx_valid : std_logic;
    signal rx_byte  : std_logic_vector(7 downto 0);
    signal tx_start     : std_logic := '0';
    signal tx_byte      : std_logic_vector(7 downto 0) := (others => '0');
    signal tx_busy      : std_logic;
    signal tx_wait_busy : std_logic := '0';

    signal packet_active : std_logic := '0';
    signal payload_index : integer range 0 to 16 := 0;
    signal request_xor    : std_logic_vector(7 downto 0) := (others => '0');

    signal blackhat_peak_reg     : std_logic_vector(7 downto 0) := (others => '0');
    signal blackhat_mean_reg     : std_logic_vector(15 downto 0) := (others => '0');
    signal threshold_reg         : std_logic_vector(15 downto 0) := (others => '0');
    signal pixel_area_reg        : std_logic_vector(7 downto 0) := (others => '0');
    signal bbox_width_reg        : std_logic_vector(7 downto 0) := (others => '0');
    signal bbox_height_reg       : std_logic_vector(7 downto 0) := (others => '0');
    signal circularity_reg       : std_logic_vector(15 downto 0) := (others => '0');
    signal solidity_reg          : std_logic_vector(15 downto 0) := (others => '0');
    signal extent_reg            : std_logic_vector(15 downto 0) := (others => '0');
    signal radial_reg            : std_logic_vector(15 downto 0) := (others => '0');

    signal gate_decision : std_logic;
    signal gate_reasons  : std_logic_vector(9 downto 0);

    signal response_active   : std_logic := '0';
    signal response_index    : integer range 0 to 4 := 0;
    signal response_decision : std_logic := '0';
    signal response_reasons  : std_logic_vector(9 downto 0) := (others => '0');
    signal response_xor      : std_logic_vector(7 downto 0) := (others => '0');
    signal last_decision     : std_logic := '0';
    signal error_latched     : std_logic := '0';
begin
    mmcm_inst : MMCME2_BASE
        generic map (
            BANDWIDTH        => "OPTIMIZED",
            CLKFBOUT_MULT_F  => 62.5,
            CLKIN1_PERIOD    => 83.333,
            CLKOUT0_DIVIDE_F => 9.375,
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

    rx_inst : entity work.uart_rx
        generic map (CLKS_PER_BIT => 80)
        port map (
            clk       => clk100,
            rx_serial => uart_txd_in,
            rx_valid  => rx_valid,
            rx_byte   => rx_byte
        );

    tx_inst : entity work.uart_tx
        generic map (CLKS_PER_BIT => 80)
        port map (
            clk       => clk100,
            tx_start  => tx_start,
            tx_byte   => tx_byte,
            tx_serial => uart_rxd_out,
            tx_busy   => tx_busy
        );

    gate_inst : entity work.bright_round_feature_gate
        port map (
            blackhat_peak     => blackhat_peak_reg,
            blackhat_mean_q8  => blackhat_mean_reg,
            threshold_q8      => threshold_reg,
            pixel_area        => pixel_area_reg,
            bbox_width        => bbox_width_reg,
            bbox_height       => bbox_height_reg,
            circularity_milli => circularity_reg,
            solidity_milli    => solidity_reg,
            extent_milli      => extent_reg,
            radial_milli      => radial_reg,
            decision          => gate_decision,
            reason_mask       => gate_reasons
        );

    process (clk100)
        variable decision_byte : std_logic_vector(7 downto 0);
        variable reason_high   : std_logic_vector(7 downto 0);
    begin
        if rising_edge(clk100) then
            tx_start <= '0';

            if system_reset_n = '0' then
                packet_active    <= '0';
                payload_index    <= 0;
                request_xor      <= (others => '0');
                response_active  <= '0';
                response_index   <= 0;
                response_decision <= '0';
                response_reasons <= (others => '0');
                response_xor     <= (others => '0');
                tx_wait_busy     <= '0';
                last_decision    <= '0';
                error_latched    <= '0';
            else
                if rx_valid = '1' then
                    if packet_active = '0' then
                        if rx_byte = REQUEST_MAGIC then
                            packet_active <= '1';
                            payload_index <= 0;
                            request_xor   <= REQUEST_MAGIC;
                        end if;
                    elsif payload_index < 16 then
                        case payload_index is
                            when 0  => blackhat_peak_reg <= rx_byte;
                            when 1  => blackhat_mean_reg(7 downto 0) <= rx_byte;
                            when 2  => blackhat_mean_reg(15 downto 8) <= rx_byte;
                            when 3  => threshold_reg(7 downto 0) <= rx_byte;
                            when 4  => threshold_reg(15 downto 8) <= rx_byte;
                            when 5  => pixel_area_reg <= rx_byte;
                            when 6  => bbox_width_reg <= rx_byte;
                            when 7  => bbox_height_reg <= rx_byte;
                            when 8  => circularity_reg(7 downto 0) <= rx_byte;
                            when 9  => circularity_reg(15 downto 8) <= rx_byte;
                            when 10 => solidity_reg(7 downto 0) <= rx_byte;
                            when 11 => solidity_reg(15 downto 8) <= rx_byte;
                            when 12 => extent_reg(7 downto 0) <= rx_byte;
                            when 13 => extent_reg(15 downto 8) <= rx_byte;
                            when 14 => radial_reg(7 downto 0) <= rx_byte;
                            when 15 => radial_reg(15 downto 8) <= rx_byte;
                            when others => null;
                        end case;
                        request_xor   <= request_xor xor rx_byte;
                        payload_index <= payload_index + 1;
                    else
                        packet_active <= '0';
                        payload_index <= 0;
                        if rx_byte = request_xor then
                            decision_byte := (others => '0');
                            decision_byte(0) := gate_decision;
                            reason_high := (others => '0');
                            reason_high(1 downto 0) := gate_reasons(9 downto 8);
                            response_decision <= gate_decision;
                            response_reasons  <= gate_reasons;
                            response_xor <= RESPONSE_MAGIC xor decision_byte xor
                                gate_reasons(7 downto 0) xor reason_high;
                            last_decision <= gate_decision;
                        else
                            response_decision <= '0';
                            response_reasons  <= (others => '1');
                            response_xor      <= x"A6";
                            last_decision     <= '0';
                            error_latched     <= '1';
                        end if;
                        response_index  <= 0;
                        response_active <= '1';
                    end if;
                end if;

                if tx_wait_busy = '1' then
                    if tx_busy = '1' then
                        tx_wait_busy <= '0';
                    end if;
                elsif response_active = '1' and tx_busy = '0' then
                    case response_index is
                        when 0 => tx_byte <= RESPONSE_MAGIC;
                        when 1 =>
                            tx_byte <= (7 downto 1 => '0') & response_decision;
                        when 2 => tx_byte <= response_reasons(7 downto 0);
                        when 3 => tx_byte <= "000000" & response_reasons(9 downto 8);
                        when 4 => tx_byte <= response_xor;
                        when others => tx_byte <= (others => '0');
                    end case;
                    tx_start     <= '1';
                    tx_wait_busy <= '1';
                    if response_index = 4 then
                        response_active <= '0';
                    else
                        response_index <= response_index + 1;
                    end if;
                end if;
            end if;
        end if;
    end process;

    led(0) <= system_reset_n;
    led(1) <= packet_active or response_active;
    led(2) <= last_decision;
    led(3) <= error_latched;
end architecture rtl;
