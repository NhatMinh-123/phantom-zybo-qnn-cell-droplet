library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity axis_int24_sum16 is
    port (
        clk     : in  std_logic;
        reset_n : in  std_logic;

        s_axis_tdata  : in  std_logic_vector(23 downto 0);
        s_axis_tvalid : in  std_logic;
        s_axis_tready : out std_logic;

        m_axis_tdata  : out std_logic_vector(31 downto 0);
        m_axis_tvalid : out std_logic;
        m_axis_tready : in  std_logic
    );
end entity axis_int24_sum16;

architecture rtl of axis_int24_sum16 is
    signal accumulator     : signed(31 downto 0) := (others => '0');
    signal sample_count    : natural range 0 to 15 := 0;
    signal output_data     : std_logic_vector(31 downto 0) := (others => '0');
    signal output_valid    : std_logic := '0';
    signal input_ready     : std_logic;
begin
    input_ready <= not output_valid;
    s_axis_tready <= input_ready;
    m_axis_tdata <= output_data;
    m_axis_tvalid <= output_valid;

    process (clk)
        variable next_sum : signed(31 downto 0);
    begin
        if rising_edge(clk) then
            if reset_n = '0' then
                accumulator <= (others => '0');
                sample_count <= 0;
                output_data  <= (others => '0');
                output_valid <= '0';
            else
                if output_valid = '1' and m_axis_tready = '1' then
                    output_valid <= '0';
                end if;

                if s_axis_tvalid = '1' and input_ready = '1' then
                    next_sum := accumulator + resize(signed(s_axis_tdata), 32);
                    if sample_count = 15 then
                        output_data  <= std_logic_vector(next_sum);
                        output_valid <= '1';
                        accumulator <= (others => '0');
                        sample_count <= 0;
                    else
                        accumulator <= next_sum;
                        sample_count <= sample_count + 1;
                    end if;
                end if;
            end if;
        end if;
    end process;
end architecture rtl;
