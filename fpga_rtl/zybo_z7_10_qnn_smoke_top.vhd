library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Minimal PL-only smoke test for the Digilent Zybo Z7-10.
-- LED[0] blinks while the remaining LEDs expose upper counter bits.
entity zybo_z7_10_qnn_smoke_top is
    port (
        sysclk : in std_logic;
        led    : out std_logic_vector(3 downto 0)
    );
end entity zybo_z7_10_qnn_smoke_top;

architecture rtl of zybo_z7_10_qnn_smoke_top is
    signal counter : unsigned(27 downto 0) := (others => '0');
begin
    process (sysclk)
    begin
        if rising_edge(sysclk) then
            counter <= counter + 1;
        end if;
    end process;

    led(0) <= counter(25);
    led(1) <= counter(24);
    led(2) <= counter(23);
    led(3) <= counter(22);
end architecture rtl;
