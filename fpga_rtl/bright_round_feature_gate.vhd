library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity bright_round_feature_gate is
    port (
        blackhat_peak     : in  std_logic_vector(7 downto 0);
        blackhat_mean_q8  : in  std_logic_vector(15 downto 0);
        threshold_q8      : in  std_logic_vector(15 downto 0);
        pixel_area        : in  std_logic_vector(7 downto 0);
        bbox_width        : in  std_logic_vector(7 downto 0);
        bbox_height       : in  std_logic_vector(7 downto 0);
        circularity_milli : in  std_logic_vector(15 downto 0);
        solidity_milli    : in  std_logic_vector(15 downto 0);
        extent_milli      : in  std_logic_vector(15 downto 0);
        radial_milli      : in  std_logic_vector(15 downto 0);
        decision          : out std_logic;
        reason_mask       : out std_logic_vector(9 downto 0)
    );
end entity bright_round_feature_gate;

architecture rtl of bright_round_feature_gate is
    attribute use_dsp : string;
    attribute use_dsp of rtl : architecture is "no";
begin
    process (all)
        variable reason_v       : std_logic_vector(9 downto 0);
        variable peak_i         : integer;
        variable mean_q_i       : integer;
        variable threshold_q_i  : integer;
        variable area_i         : integer;
        variable width_i        : integer;
        variable height_i       : integer;
        variable min_side_i     : integer;
        variable max_side_i     : integer;
        variable circularity_i  : integer;
        variable solidity_i     : integer;
        variable extent_i       : integer;
        variable radial_i       : integer;
    begin
        peak_i        := to_integer(unsigned(blackhat_peak));
        mean_q_i      := to_integer(unsigned(blackhat_mean_q8));
        threshold_q_i := to_integer(unsigned(threshold_q8));
        area_i        := to_integer(unsigned(pixel_area));
        width_i       := to_integer(unsigned(bbox_width));
        height_i      := to_integer(unsigned(bbox_height));
        circularity_i := to_integer(unsigned(circularity_milli));
        solidity_i    := to_integer(unsigned(solidity_milli));
        extent_i      := to_integer(unsigned(extent_milli));
        radial_i      := to_integer(unsigned(radial_milli));

        if width_i < height_i then
            min_side_i := width_i;
            max_side_i := height_i;
        else
            min_side_i := height_i;
            max_side_i := width_i;
        end if;

        reason_v := (others => '0');

        -- Bit order matches scripts/bright_round_feature_gate.py.
        if peak_i < 20 then
            reason_v(0) := '1';
        end if;
        if mean_q_i < 8 * 256 then
            reason_v(1) := '1';
        end if;
        if peak_i * 512 < threshold_q_i * 5 then
            reason_v(2) := '1';
        end if;
        if mean_q_i * 2 < threshold_q_i * 3 then
            reason_v(3) := '1';
        end if;
        if area_i < 4 or area_i > 100 then
            reason_v(4) := '1';
        end if;
        if max_side_i = 0 or min_side_i * 20 < max_side_i * 11 then
            reason_v(5) := '1';
        end if;
        if circularity_i < 420 then
            reason_v(6) := '1';
        end if;
        if solidity_i < 550 then
            reason_v(7) := '1';
        end if;
        if extent_i < 500 then
            reason_v(8) := '1';
        end if;
        if radial_i > 900 then
            reason_v(9) := '1';
        end if;

        reason_mask <= reason_v;
        if reason_v = "0000000000" then
            decision <= '1';
        else
            decision <= '0';
        end if;
    end process;
end architecture rtl;
