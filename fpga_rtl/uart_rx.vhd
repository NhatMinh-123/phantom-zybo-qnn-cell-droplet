library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity uart_rx is
    generic (
        CLKS_PER_BIT : positive := 104
    );
    port (
        clk       : in  std_logic;
        rx_serial : in  std_logic;
        rx_valid  : out std_logic;
        rx_byte   : out std_logic_vector(7 downto 0)
    );
end entity uart_rx;

architecture rtl of uart_rx is
    type state_t is (idle, start_bit, data_bits, stop_bit, cleanup);

    signal state       : state_t := idle;
    signal rx_sync_1   : std_logic := '1';
    signal rx_sync_2   : std_logic := '1';
    signal clock_count : integer range 0 to CLKS_PER_BIT - 1 := 0;
    signal bit_index   : integer range 0 to 7 := 0;
    signal data_reg    : std_logic_vector(7 downto 0) := (others => '0');
begin
    process (clk)
    begin
        if rising_edge(clk) then
            rx_sync_1 <= rx_serial;
            rx_sync_2 <= rx_sync_1;
        end if;
    end process;

    process (clk)
    begin
        if rising_edge(clk) then
            rx_valid <= '0';

            case state is
                when idle =>
                    clock_count <= 0;
                    bit_index   <= 0;

                    if rx_sync_2 = '0' then
                        state <= start_bit;
                    end if;

                when start_bit =>
                    if clock_count = (CLKS_PER_BIT - 1) / 2 then
                        if rx_sync_2 = '0' then
                            clock_count <= 0;
                            state       <= data_bits;
                        else
                            state <= idle;
                        end if;
                    else
                        clock_count <= clock_count + 1;
                    end if;

                when data_bits =>
                    if clock_count = CLKS_PER_BIT - 1 then
                        clock_count         <= 0;
                        data_reg(bit_index) <= rx_sync_2;

                        if bit_index = 7 then
                            bit_index <= 0;
                            state     <= stop_bit;
                        else
                            bit_index <= bit_index + 1;
                        end if;
                    else
                        clock_count <= clock_count + 1;
                    end if;

                when stop_bit =>
                    if clock_count = CLKS_PER_BIT - 1 then
                        rx_byte     <= data_reg;
                        rx_valid    <= '1';
                        clock_count <= 0;
                        state       <= cleanup;
                    else
                        clock_count <= clock_count + 1;
                    end if;

                when cleanup =>
                    state <= idle;
            end case;
        end if;
    end process;
end architecture rtl;
