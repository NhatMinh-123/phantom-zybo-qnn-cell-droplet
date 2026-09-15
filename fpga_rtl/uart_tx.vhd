library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity uart_tx is
    generic (
        CLKS_PER_BIT : positive := 104
    );
    port (
        clk       : in  std_logic;
        tx_start  : in  std_logic;
        tx_byte   : in  std_logic_vector(7 downto 0);
        tx_serial : out std_logic;
        tx_busy   : out std_logic
    );
end entity uart_tx;

architecture rtl of uart_tx is
    type state_t is (idle, start_bit, data_bits, stop_bit, cleanup);

    signal state       : state_t := idle;
    signal clock_count : integer range 0 to CLKS_PER_BIT - 1 := 0;
    signal bit_index   : integer range 0 to 7 := 0;
    signal data_reg    : std_logic_vector(7 downto 0) := (others => '0');
    signal tx_reg      : std_logic := '1';
    signal busy_reg    : std_logic := '0';
begin
    tx_serial <= tx_reg;
    tx_busy   <= busy_reg;

    process (clk)
    begin
        if rising_edge(clk) then
            case state is
                when idle =>
                    tx_reg      <= '1';
                    busy_reg    <= '0';
                    clock_count <= 0;
                    bit_index   <= 0;

                    if tx_start = '1' then
                        data_reg <= tx_byte;
                        busy_reg <= '1';
                        state    <= start_bit;
                    end if;

                when start_bit =>
                    tx_reg <= '0';
                    if clock_count = CLKS_PER_BIT - 1 then
                        clock_count <= 0;
                        state       <= data_bits;
                    else
                        clock_count <= clock_count + 1;
                    end if;

                when data_bits =>
                    tx_reg <= data_reg(bit_index);

                    if clock_count = CLKS_PER_BIT - 1 then
                        clock_count <= 0;
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
                    tx_reg <= '1';
                    if clock_count = CLKS_PER_BIT - 1 then
                        clock_count <= 0;
                        state       <= cleanup;
                    else
                        clock_count <= clock_count + 1;
                    end if;

                when cleanup =>
                    busy_reg <= '0';
                    state    <= idle;
            end case;
        end if;
    end process;
end architecture rtl;
