library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.detector_output_thresholds_15micro_pkg.all;

entity detector_output_requantizer_15micro is
    generic (
        INPUT_BITS : positive := 24
    );
    port (
        clk     : in std_logic;
        reset_n : in std_logic;

        s_axis_tdata  : in  std_logic_vector(INPUT_BITS - 1 downto 0);
        s_axis_tvalid : in  std_logic;
        s_axis_tready : out std_logic;

        m_axis_tdata  : out std_logic_vector(7 downto 0);
        m_axis_tvalid : out std_logic;
        m_axis_tready : in  std_logic
    );
end entity detector_output_requantizer_15micro;

architecture rtl of detector_output_requantizer_15micro is
    type state_t is (idle, issue_read, compare_threshold, hold_output);

    signal state : state_t := idle;
    signal accumulator : signed(INPUT_BITS - 1 downto 0) := (others => '0');
    signal low_bound  : natural range 0 to 255 := 0;
    signal high_bound : natural range 0 to 255 := 255;
    signal midpoint   : natural range 0 to 254 := 127;
    signal iteration  : natural range 0 to 7 := 0;
    signal channel_index : natural range 0 to DETECTOR_OUTPUT_CHANNELS - 1 := 0;
    signal channel_base : natural range 0 to
        (DETECTOR_OUTPUT_CHANNELS - 1) * DETECTOR_THRESHOLDS_PER_CHANNEL := 0;

    signal threshold_rom : detector_threshold_rom_t := DETECTOR_OUTPUT_THRESHOLDS;
    signal threshold_data : signed(DETECTOR_THRESHOLD_BITS - 1 downto 0);
    signal output_data_reg : std_logic_vector(7 downto 0) := (others => '0');

    attribute rom_style : string;
    attribute rom_style of threshold_rom : signal is "block";
begin
    s_axis_tready <= '1' when state = idle else '0';
    m_axis_tvalid <= '1' when state = hold_output else '0';
    m_axis_tdata <= output_data_reg;

    process (clk)
    begin
        if rising_edge(clk) then
            if state = issue_read then
                threshold_data <= threshold_rom(channel_base + midpoint);
            end if;
        end if;
    end process;

    process (clk)
        variable next_low  : natural range 0 to 255;
        variable next_high : natural range 0 to 255;
        variable next_midpoint : natural range 0 to 254;
    begin
        if rising_edge(clk) then
            if reset_n = '0' then
                state <= idle;
                accumulator <= (others => '0');
                low_bound <= 0;
                high_bound <= 255;
                midpoint <= 127;
                iteration <= 0;
                channel_index <= 0;
                channel_base <= 0;
                output_data_reg <= (others => '0');
            else
                case state is
                    when idle =>
                        if s_axis_tvalid = '1' then
                            accumulator <= signed(s_axis_tdata);
                            low_bound <= 0;
                            high_bound <= 255;
                            midpoint <= 127;
                            iteration <= 0;
                            state <= issue_read;
                        end if;

                    when issue_read =>
                        state <= compare_threshold;

                    when compare_threshold =>
                        next_low := low_bound;
                        next_high := high_bound;
                        if resize(threshold_data, INPUT_BITS) <= accumulator then
                            next_low := midpoint + 1;
                        else
                            next_high := midpoint;
                        end if;

                        if iteration = 7 then
                            output_data_reg <= std_logic_vector(
                                to_signed(integer(next_low) - 128, 8)
                            );
                            state <= hold_output;
                        else
                            low_bound <= next_low;
                            high_bound <= next_high;
                            next_midpoint := (next_low + next_high) / 2;
                            midpoint <= next_midpoint;
                            iteration <= iteration + 1;
                            state <= issue_read;
                        end if;

                    when hold_output =>
                        if m_axis_tready = '1' then
                            if channel_index = DETECTOR_OUTPUT_CHANNELS - 1 then
                                channel_index <= 0;
                                channel_base <= 0;
                            else
                                channel_index <= channel_index + 1;
                                channel_base <=
                                    channel_base + DETECTOR_THRESHOLDS_PER_CHANNEL;
                            end if;
                            state <= idle;
                        end if;
                end case;
            end if;
        end if;
    end process;
end architecture rtl;
