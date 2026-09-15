library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity classical_roi_gate_analyzer is
    generic (
        IMAGE_SIZE                 : positive := 96;
        GATE_X                     : natural := 48;
        DROPLET_SCORE_THRESHOLD    : natural := 863;
        CELL_RESPONSE_THRESHOLD_X8 : natural := 320;
        INPUT_RGB332               : boolean := false
    );
    port (
        clk     : in std_logic;
        reset_n : in std_logic;

        s_axis_tdata  : in  std_logic_vector(7 downto 0);
        s_axis_tvalid : in  std_logic;
        s_axis_tready : out std_logic;

        frame_done        : out std_logic;
        droplet_score     : out std_logic_vector(15 downto 0);
        droplet_center_y  : out std_logic_vector(6 downto 0);
        droplet_radius    : out std_logic_vector(6 downto 0);
        droplet_present   : out std_logic;
        cell_count        : out std_logic_vector(1 downto 0);
        cell0_x           : out std_logic_vector(6 downto 0);
        cell0_y           : out std_logic_vector(6 downto 0);
        cell0_score       : out std_logic_vector(7 downto 0);
        cell1_x           : out std_logic_vector(6 downto 0);
        cell1_y           : out std_logic_vector(6 downto 0);
        cell1_score       : out std_logic_vector(7 downto 0)
    );
end entity classical_roi_gate_analyzer;

architecture rtl of classical_roi_gate_analyzer is
    constant IMAGE_PIXELS       : positive := IMAGE_SIZE * IMAGE_SIZE;
    constant RADIUS_COUNT       : positive := 9;
    constant DIRECTION_COUNT    : positive := 16;
    constant RADIAL_DELTA       : positive := 4;
    constant SUPPORT_THRESHOLD  : integer := 4;
    constant SUPPORT_BONUS      : integer := 8;
    constant CELL_X_MIN         : natural := GATE_X - 5;
    constant CELL_X_MAX         : natural := GATE_X + 5;
    constant CELL_Y_MIN         : natural := 3;
    constant CELL_Y_MAX         : natural := IMAGE_SIZE - 4;
    constant CELL_MIN_Y_DISTANCE : natural := 6;

    type image_memory_t is array (0 to IMAGE_PIXELS - 1) of
        std_logic_vector(15 downto 0);
    type state_t is (
        collect_image,
        ring_sample_prepare,
        ring_offset_compute,
        ring_coordinate_compute,
        ring_row_compute,
        ring_address_compute,
        ring_issue,
        ring_wait,
        ring_consume,
        ring_contrast_compute,
        ring_accumulate,
        ring_finalize_score,
        ring_compare_best,
        ring_advance,
        cell_sample_prepare,
        cell_row_compute,
        cell_address_compute,
        cell_issue,
        cell_wait,
        cell_consume,
        cell_candidate_update,
        cell_scan_advance,
        hold_result
    );

    signal feature_image_memory : image_memory_t;
    signal state : state_t := collect_image;
    signal write_address : natural range 0 to IMAGE_PIXELS - 1 := 0;
    signal write_count : natural range 0 to IMAGE_PIXELS := 0;
    signal read_address : natural range 0 to IMAGE_PIXELS - 1 := 0;
    signal feature_read_data : std_logic_vector(15 downto 0) :=
        (others => '0');

    signal radius_index : natural range 0 to RADIUS_COUNT - 1 := 0;
    signal ring_center_y : natural range 0 to IMAGE_SIZE - 1 := 22;
    signal direction_index : natural range 0 to DIRECTION_COUNT - 1 := 0;
    signal ring_phase : natural range 0 to 2 := 0;
    signal sample_radius_reg : natural range 0 to IMAGE_SIZE - 1 := 0;
    signal sample_direction_x_reg : integer range -1024 to 1024 := 0;
    signal sample_direction_y_reg : integer range -1024 to 1024 := 0;
    signal sample_offset_x_reg : integer range -IMAGE_SIZE to IMAGE_SIZE := 0;
    signal sample_offset_y_reg : integer range -IMAGE_SIZE to IMAGE_SIZE := 0;
    signal sample_x_reg : natural range 0 to IMAGE_SIZE - 1 := 0;
    signal sample_y_reg : natural range 0 to IMAGE_SIZE - 1 := 0;
    signal sample_row_base_reg : natural range 0 to IMAGE_PIXELS - 1 := 0;
    signal ring_inner : integer range 0 to 255 := 0;
    signal ring_value : integer range 0 to 255 := 0;
    signal ring_outer : integer range 0 to 255 := 0;
    signal ring_contrast_reg : integer range -510 to 510 := 0;
    signal ring_score_acc : integer range 0 to 32767 := 0;
    signal ring_support_acc : natural range 0 to DIRECTION_COUNT := 0;
    signal completed_ring_score : natural range 0 to 32767 := 0;
    signal completed_ring_support : natural range 0 to DIRECTION_COUNT := 0;
    signal final_ring_score_reg : natural range 0 to 65535 := 0;
    signal best_ring_score : natural range 0 to 65535 := 0;
    signal best_ring_y : natural range 0 to IMAGE_SIZE - 1 := IMAGE_SIZE / 2;
    signal best_ring_radius : natural range 0 to IMAGE_SIZE - 1 := 18;

    signal cell_x_scan : natural range CELL_X_MIN to CELL_X_MAX := CELL_X_MIN;
    signal cell_y_scan : natural range CELL_Y_MIN to CELL_Y_MAX := CELL_Y_MIN;
    signal cell_sample_index : natural range 0 to 8 := 0;
    signal cell_accumulator : integer range -4096 to 4095 := 0;
    signal cell_response_reg : integer range -4096 to 4095 := 0;
    signal top0_valid : std_logic := '0';
    signal top1_valid : std_logic := '0';
    signal top0_x : natural range 0 to IMAGE_SIZE - 1 := GATE_X;
    signal top1_x : natural range 0 to IMAGE_SIZE - 1 := GATE_X;
    signal top0_y : natural range 0 to IMAGE_SIZE - 1 := 0;
    signal top1_y : natural range 0 to IMAGE_SIZE - 1 := 0;
    signal top0_score : natural range 0 to 255 := 0;
    signal top1_score : natural range 0 to 255 := 0;

    attribute ram_style : string;
    attribute ram_style of feature_image_memory : signal is "block";

    function direction_x_q10(index : natural) return integer is
    begin
        case index is
            when 0  => return 1024;
            when 1  => return 946;
            when 2  => return 724;
            when 3  => return 392;
            when 4  => return 0;
            when 5  => return -392;
            when 6  => return -724;
            when 7  => return -946;
            when 8  => return -1024;
            when 9  => return -946;
            when 10 => return -724;
            when 11 => return -392;
            when 12 => return 0;
            when 13 => return 392;
            when 14 => return 724;
            when others => return 946;
        end case;
    end function;

    function direction_y_q10(index : natural) return integer is
    begin
        case index is
            when 0  => return 0;
            when 1  => return 392;
            when 2  => return 724;
            when 3  => return 946;
            when 4  => return 1024;
            when 5  => return 946;
            when 6  => return 724;
            when 7  => return 392;
            when 8  => return 0;
            when 9  => return -392;
            when 10 => return -724;
            when 11 => return -946;
            when 12 => return -1024;
            when 13 => return -946;
            when 14 => return -724;
            when others => return -392;
        end case;
    end function;

    function scale_q10(value : integer; direction : integer) return integer is
        variable product : integer;
    begin
        product := value * direction;
        if product >= 0 then
            return (product + 512) / 1024;
        end if;
        return -((-product + 512) / 1024);
    end function;

    function address_of(x : integer; y : integer) return natural is
    begin
        return natural(y * IMAGE_SIZE + x);
    end function;

    function abs_difference(left_value : integer; right_value : integer)
        return natural is
    begin
        if left_value >= right_value then
            return natural(left_value - right_value);
        end if;
        return natural(right_value - left_value);
    end function;

    -- Explicit 8-bit ROMs keep the RGB332 conversion narrow in synthesis.
    function rgb332_cb(value : std_logic_vector(7 downto 0))
        return std_logic_vector is
    begin
        case to_integer(unsigned(value)) is
            when 0 => return x"80";
            when 1 => return x"AA";
            when 2 => return x"D4";
            when 3 => return x"FE";
            when 4 => return x"74";
            when 5 => return x"9E";
            when 6 => return x"C8";
            when 7 => return x"F2";
            when 8 => return x"68";
            when 9 => return x"92";
            when 10 => return x"BC";
            when 11 => return x"E6";
            when 12 => return x"5C";
            when 13 => return x"86";
            when 14 => return x"B0";
            when 15 => return x"DA";
            when 16 => return x"50";
            when 17 => return x"7A";
            when 18 => return x"A4";
            when 19 => return x"CE";
            when 20 => return x"44";
            when 21 => return x"6E";
            when 22 => return x"98";
            when 23 => return x"C2";
            when 24 => return x"38";
            when 25 => return x"62";
            when 26 => return x"8C";
            when 27 => return x"B6";
            when 28 => return x"2C";
            when 29 => return x"56";
            when 30 => return x"80";
            when 31 => return x"AA";
            when 32 => return x"7A";
            when 33 => return x"A4";
            when 34 => return x"CE";
            when 35 => return x"F8";
            when 36 => return x"6E";
            when 37 => return x"98";
            when 38 => return x"C2";
            when 39 => return x"EC";
            when 40 => return x"62";
            when 41 => return x"8C";
            when 42 => return x"B6";
            when 43 => return x"E0";
            when 44 => return x"56";
            when 45 => return x"80";
            when 46 => return x"AA";
            when 47 => return x"D4";
            when 48 => return x"4A";
            when 49 => return x"74";
            when 50 => return x"9E";
            when 51 => return x"C8";
            when 52 => return x"3E";
            when 53 => return x"68";
            when 54 => return x"92";
            when 55 => return x"BC";
            when 56 => return x"32";
            when 57 => return x"5C";
            when 58 => return x"86";
            when 59 => return x"B0";
            when 60 => return x"26";
            when 61 => return x"50";
            when 62 => return x"7A";
            when 63 => return x"A4";
            when 64 => return x"74";
            when 65 => return x"9E";
            when 66 => return x"C8";
            when 67 => return x"F2";
            when 68 => return x"68";
            when 69 => return x"92";
            when 70 => return x"BC";
            when 71 => return x"E6";
            when 72 => return x"5C";
            when 73 => return x"86";
            when 74 => return x"B0";
            when 75 => return x"DA";
            when 76 => return x"50";
            when 77 => return x"7A";
            when 78 => return x"A4";
            when 79 => return x"CE";
            when 80 => return x"44";
            when 81 => return x"6E";
            when 82 => return x"98";
            when 83 => return x"C2";
            when 84 => return x"38";
            when 85 => return x"62";
            when 86 => return x"8C";
            when 87 => return x"B6";
            when 88 => return x"2C";
            when 89 => return x"56";
            when 90 => return x"80";
            when 91 => return x"AA";
            when 92 => return x"20";
            when 93 => return x"4A";
            when 94 => return x"74";
            when 95 => return x"9E";
            when 96 => return x"6E";
            when 97 => return x"98";
            when 98 => return x"C2";
            when 99 => return x"EC";
            when 100 => return x"62";
            when 101 => return x"8C";
            when 102 => return x"B6";
            when 103 => return x"E0";
            when 104 => return x"56";
            when 105 => return x"80";
            when 106 => return x"AA";
            when 107 => return x"D4";
            when 108 => return x"4A";
            when 109 => return x"74";
            when 110 => return x"9E";
            when 111 => return x"C8";
            when 112 => return x"3E";
            when 113 => return x"68";
            when 114 => return x"92";
            when 115 => return x"BC";
            when 116 => return x"32";
            when 117 => return x"5C";
            when 118 => return x"86";
            when 119 => return x"B0";
            when 120 => return x"26";
            when 121 => return x"50";
            when 122 => return x"7A";
            when 123 => return x"A4";
            when 124 => return x"1A";
            when 125 => return x"44";
            when 126 => return x"6E";
            when 127 => return x"98";
            when 128 => return x"68";
            when 129 => return x"92";
            when 130 => return x"BC";
            when 131 => return x"E6";
            when 132 => return x"5C";
            when 133 => return x"86";
            when 134 => return x"B0";
            when 135 => return x"DA";
            when 136 => return x"50";
            when 137 => return x"7A";
            when 138 => return x"A4";
            when 139 => return x"CE";
            when 140 => return x"44";
            when 141 => return x"6E";
            when 142 => return x"98";
            when 143 => return x"C2";
            when 144 => return x"38";
            when 145 => return x"62";
            when 146 => return x"8C";
            when 147 => return x"B6";
            when 148 => return x"2C";
            when 149 => return x"56";
            when 150 => return x"80";
            when 151 => return x"AA";
            when 152 => return x"20";
            when 153 => return x"4A";
            when 154 => return x"74";
            when 155 => return x"9E";
            when 156 => return x"14";
            when 157 => return x"3E";
            when 158 => return x"68";
            when 159 => return x"92";
            when 160 => return x"62";
            when 161 => return x"8C";
            when 162 => return x"B6";
            when 163 => return x"E0";
            when 164 => return x"56";
            when 165 => return x"80";
            when 166 => return x"AA";
            when 167 => return x"D4";
            when 168 => return x"4A";
            when 169 => return x"74";
            when 170 => return x"9E";
            when 171 => return x"C8";
            when 172 => return x"3E";
            when 173 => return x"68";
            when 174 => return x"92";
            when 175 => return x"BC";
            when 176 => return x"32";
            when 177 => return x"5C";
            when 178 => return x"86";
            when 179 => return x"B0";
            when 180 => return x"26";
            when 181 => return x"50";
            when 182 => return x"7A";
            when 183 => return x"A4";
            when 184 => return x"1A";
            when 185 => return x"44";
            when 186 => return x"6E";
            when 187 => return x"98";
            when 188 => return x"0E";
            when 189 => return x"38";
            when 190 => return x"62";
            when 191 => return x"8C";
            when 192 => return x"5C";
            when 193 => return x"86";
            when 194 => return x"B0";
            when 195 => return x"DA";
            when 196 => return x"50";
            when 197 => return x"7A";
            when 198 => return x"A4";
            when 199 => return x"CE";
            when 200 => return x"44";
            when 201 => return x"6E";
            when 202 => return x"98";
            when 203 => return x"C2";
            when 204 => return x"38";
            when 205 => return x"62";
            when 206 => return x"8C";
            when 207 => return x"B6";
            when 208 => return x"2C";
            when 209 => return x"56";
            when 210 => return x"80";
            when 211 => return x"AA";
            when 212 => return x"20";
            when 213 => return x"4A";
            when 214 => return x"74";
            when 215 => return x"9E";
            when 216 => return x"14";
            when 217 => return x"3E";
            when 218 => return x"68";
            when 219 => return x"92";
            when 220 => return x"08";
            when 221 => return x"32";
            when 222 => return x"5C";
            when 223 => return x"86";
            when 224 => return x"56";
            when 225 => return x"80";
            when 226 => return x"AA";
            when 227 => return x"D4";
            when 228 => return x"4A";
            when 229 => return x"74";
            when 230 => return x"9E";
            when 231 => return x"C8";
            when 232 => return x"3E";
            when 233 => return x"68";
            when 234 => return x"92";
            when 235 => return x"BC";
            when 236 => return x"32";
            when 237 => return x"5C";
            when 238 => return x"86";
            when 239 => return x"B0";
            when 240 => return x"26";
            when 241 => return x"50";
            when 242 => return x"7A";
            when 243 => return x"A4";
            when 244 => return x"1A";
            when 245 => return x"44";
            when 246 => return x"6E";
            when 247 => return x"98";
            when 248 => return x"0E";
            when 249 => return x"38";
            when 250 => return x"62";
            when 251 => return x"8C";
            when 252 => return x"02";
            when 253 => return x"2C";
            when 254 => return x"56";
            when 255 => return x"80";
            when others => return x"80";
        end case;
    end function;

    function rgb332_luma(value : std_logic_vector(7 downto 0))
        return std_logic_vector is
    begin
        case to_integer(unsigned(value)) is
            when 0 => return x"00";
            when 1 => return x"15";
            when 2 => return x"2A";
            when 3 => return x"3F";
            when 4 => return x"12";
            when 5 => return x"27";
            when 6 => return x"3C";
            when 7 => return x"51";
            when 8 => return x"24";
            when 9 => return x"39";
            when 10 => return x"4E";
            when 11 => return x"63";
            when 12 => return x"36";
            when 13 => return x"4B";
            when 14 => return x"60";
            when 15 => return x"75";
            when 16 => return x"48";
            when 17 => return x"5D";
            when 18 => return x"72";
            when 19 => return x"87";
            when 20 => return x"5A";
            when 21 => return x"6F";
            when 22 => return x"84";
            when 23 => return x"99";
            when 24 => return x"6C";
            when 25 => return x"81";
            when 26 => return x"96";
            when 27 => return x"AB";
            when 28 => return x"7E";
            when 29 => return x"93";
            when 30 => return x"A8";
            when 31 => return x"BD";
            when 32 => return x"09";
            when 33 => return x"1E";
            when 34 => return x"33";
            when 35 => return x"48";
            when 36 => return x"1B";
            when 37 => return x"30";
            when 38 => return x"45";
            when 39 => return x"5A";
            when 40 => return x"2D";
            when 41 => return x"42";
            when 42 => return x"57";
            when 43 => return x"6C";
            when 44 => return x"3F";
            when 45 => return x"54";
            when 46 => return x"69";
            when 47 => return x"7E";
            when 48 => return x"51";
            when 49 => return x"66";
            when 50 => return x"7B";
            when 51 => return x"90";
            when 52 => return x"63";
            when 53 => return x"78";
            when 54 => return x"8D";
            when 55 => return x"A2";
            when 56 => return x"75";
            when 57 => return x"8A";
            when 58 => return x"9F";
            when 59 => return x"B4";
            when 60 => return x"87";
            when 61 => return x"9C";
            when 62 => return x"B1";
            when 63 => return x"C6";
            when 64 => return x"12";
            when 65 => return x"27";
            when 66 => return x"3C";
            when 67 => return x"51";
            when 68 => return x"24";
            when 69 => return x"39";
            when 70 => return x"4E";
            when 71 => return x"63";
            when 72 => return x"36";
            when 73 => return x"4B";
            when 74 => return x"60";
            when 75 => return x"75";
            when 76 => return x"48";
            when 77 => return x"5D";
            when 78 => return x"72";
            when 79 => return x"87";
            when 80 => return x"5A";
            when 81 => return x"6F";
            when 82 => return x"84";
            when 83 => return x"99";
            when 84 => return x"6C";
            when 85 => return x"81";
            when 86 => return x"96";
            when 87 => return x"AB";
            when 88 => return x"7E";
            when 89 => return x"93";
            when 90 => return x"A8";
            when 91 => return x"BD";
            when 92 => return x"90";
            when 93 => return x"A5";
            when 94 => return x"BA";
            when 95 => return x"CF";
            when 96 => return x"1B";
            when 97 => return x"30";
            when 98 => return x"45";
            when 99 => return x"5A";
            when 100 => return x"2D";
            when 101 => return x"42";
            when 102 => return x"57";
            when 103 => return x"6C";
            when 104 => return x"3F";
            when 105 => return x"54";
            when 106 => return x"69";
            when 107 => return x"7E";
            when 108 => return x"51";
            when 109 => return x"66";
            when 110 => return x"7B";
            when 111 => return x"90";
            when 112 => return x"63";
            when 113 => return x"78";
            when 114 => return x"8D";
            when 115 => return x"A2";
            when 116 => return x"75";
            when 117 => return x"8A";
            when 118 => return x"9F";
            when 119 => return x"B4";
            when 120 => return x"87";
            when 121 => return x"9C";
            when 122 => return x"B1";
            when 123 => return x"C6";
            when 124 => return x"99";
            when 125 => return x"AE";
            when 126 => return x"C3";
            when 127 => return x"D8";
            when 128 => return x"24";
            when 129 => return x"39";
            when 130 => return x"4E";
            when 131 => return x"63";
            when 132 => return x"36";
            when 133 => return x"4B";
            when 134 => return x"60";
            when 135 => return x"75";
            when 136 => return x"48";
            when 137 => return x"5D";
            when 138 => return x"72";
            when 139 => return x"87";
            when 140 => return x"5A";
            when 141 => return x"6F";
            when 142 => return x"84";
            when 143 => return x"99";
            when 144 => return x"6C";
            when 145 => return x"81";
            when 146 => return x"96";
            when 147 => return x"AB";
            when 148 => return x"7E";
            when 149 => return x"93";
            when 150 => return x"A8";
            when 151 => return x"BD";
            when 152 => return x"90";
            when 153 => return x"A5";
            when 154 => return x"BA";
            when 155 => return x"CF";
            when 156 => return x"A2";
            when 157 => return x"B7";
            when 158 => return x"CC";
            when 159 => return x"E1";
            when 160 => return x"2D";
            when 161 => return x"42";
            when 162 => return x"57";
            when 163 => return x"6C";
            when 164 => return x"3F";
            when 165 => return x"54";
            when 166 => return x"69";
            when 167 => return x"7E";
            when 168 => return x"51";
            when 169 => return x"66";
            when 170 => return x"7B";
            when 171 => return x"90";
            when 172 => return x"63";
            when 173 => return x"78";
            when 174 => return x"8D";
            when 175 => return x"A2";
            when 176 => return x"75";
            when 177 => return x"8A";
            when 178 => return x"9F";
            when 179 => return x"B4";
            when 180 => return x"87";
            when 181 => return x"9C";
            when 182 => return x"B1";
            when 183 => return x"C6";
            when 184 => return x"99";
            when 185 => return x"AE";
            when 186 => return x"C3";
            when 187 => return x"D8";
            when 188 => return x"AB";
            when 189 => return x"C0";
            when 190 => return x"D5";
            when 191 => return x"EA";
            when 192 => return x"36";
            when 193 => return x"4B";
            when 194 => return x"60";
            when 195 => return x"75";
            when 196 => return x"48";
            when 197 => return x"5D";
            when 198 => return x"72";
            when 199 => return x"87";
            when 200 => return x"5A";
            when 201 => return x"6F";
            when 202 => return x"84";
            when 203 => return x"99";
            when 204 => return x"6C";
            when 205 => return x"81";
            when 206 => return x"96";
            when 207 => return x"AB";
            when 208 => return x"7E";
            when 209 => return x"93";
            when 210 => return x"A8";
            when 211 => return x"BD";
            when 212 => return x"90";
            when 213 => return x"A5";
            when 214 => return x"BA";
            when 215 => return x"CF";
            when 216 => return x"A2";
            when 217 => return x"B7";
            when 218 => return x"CC";
            when 219 => return x"E1";
            when 220 => return x"B4";
            when 221 => return x"C9";
            when 222 => return x"DE";
            when 223 => return x"F3";
            when 224 => return x"3F";
            when 225 => return x"54";
            when 226 => return x"69";
            when 227 => return x"7E";
            when 228 => return x"51";
            when 229 => return x"66";
            when 230 => return x"7B";
            when 231 => return x"90";
            when 232 => return x"63";
            when 233 => return x"78";
            when 234 => return x"8D";
            when 235 => return x"A2";
            when 236 => return x"75";
            when 237 => return x"8A";
            when 238 => return x"9F";
            when 239 => return x"B4";
            when 240 => return x"87";
            when 241 => return x"9C";
            when 242 => return x"B1";
            when 243 => return x"C6";
            when 244 => return x"99";
            when 245 => return x"AE";
            when 246 => return x"C3";
            when 247 => return x"D8";
            when 248 => return x"AB";
            when 249 => return x"C0";
            when 250 => return x"D5";
            when 251 => return x"EA";
            when 252 => return x"BD";
            when 253 => return x"D2";
            when 254 => return x"E7";
            when 255 => return x"FC";
            when others => return x"00";
        end case;
    end function;

    function cell_offset_x(index : natural) return integer is
    begin
        case index is
            when 0 | 1 | 2 => return 0;
            when 3 => return -3;
            when 4 => return 3;
            when 5 | 7 => return -2;
            when others => return 2;
        end case;
    end function;

    function cell_offset_y(index : natural) return integer is
    begin
        case index is
            when 0 | 3 | 4 => return 0;
            when 1 => return -3;
            when 2 => return 3;
            when 5 | 6 => return -2;
            when others => return 2;
        end case;
    end function;
begin
    assert IMAGE_SIZE = 96
        report "Classical gate analyzer is calibrated for 96x96 input"
        severity failure;

    s_axis_tready <= '1' when state = collect_image else '0';
    frame_done <= '1' when state = hold_result else '0';
    droplet_score <= std_logic_vector(to_unsigned(best_ring_score, 16));
    droplet_center_y <= std_logic_vector(to_unsigned(best_ring_y, 7));
    droplet_radius <= std_logic_vector(to_unsigned(best_ring_radius, 7));
    droplet_present <= '1' when best_ring_score >= DROPLET_SCORE_THRESHOLD else '0';
    cell_count <= "10" when top1_valid = '1' else
                  "01" when top0_valid = '1' else "00";
    cell0_x <= std_logic_vector(to_unsigned(top0_x, 7));
    cell0_y <= std_logic_vector(to_unsigned(top0_y, 7));
    cell1_x <= std_logic_vector(to_unsigned(top1_x, 7));
    cell1_y <= std_logic_vector(to_unsigned(top1_y, 7));
    cell0_score <= std_logic_vector(to_unsigned(top0_score, 8));
    cell1_score <= std_logic_vector(to_unsigned(top1_score, 8));

    process (clk)
    begin
        if rising_edge(clk) then
            if state = collect_image and s_axis_tvalid = '1' then
                if INPUT_RGB332 then
                    feature_image_memory(write_address) <=
                        rgb332_cb(s_axis_tdata) & rgb332_luma(s_axis_tdata);
                else
                    feature_image_memory(write_address) <=
                        s_axis_tdata & s_axis_tdata;
                end if;
            end if;
            feature_read_data <= feature_image_memory(read_address);
        end if;
    end process;

    process (clk)
        variable radius_value : integer;
        variable margin_value : integer;
        variable sample_radius : integer;
        variable sample_x : integer;
        variable sample_y : integer;
        variable sample_value : integer;
        variable contrast_value : integer;
        variable score_after : integer;
        variable support_after : integer;
        variable final_score : integer;
        variable response_value : integer;
        variable clipped_response : natural;
        variable close_top0 : boolean;
        variable close_top1 : boolean;
    begin
        if rising_edge(clk) then
            if reset_n = '0' then
                state <= collect_image;
                write_address <= 0;
                write_count <= 0;
                read_address <= 0;
                radius_index <= 0;
                ring_center_y <= 22;
                direction_index <= 0;
                ring_phase <= 0;
                sample_radius_reg <= 0;
                sample_direction_x_reg <= 0;
                sample_direction_y_reg <= 0;
                sample_offset_x_reg <= 0;
                sample_offset_y_reg <= 0;
                sample_x_reg <= 0;
                sample_y_reg <= 0;
                sample_row_base_reg <= 0;
                ring_inner <= 0;
                ring_value <= 0;
                ring_outer <= 0;
                ring_contrast_reg <= 0;
                ring_score_acc <= 0;
                ring_support_acc <= 0;
                completed_ring_score <= 0;
                completed_ring_support <= 0;
                final_ring_score_reg <= 0;
                best_ring_score <= 0;
                best_ring_y <= IMAGE_SIZE / 2;
                best_ring_radius <= 18;
                cell_x_scan <= CELL_X_MIN;
                cell_y_scan <= CELL_Y_MIN;
                cell_sample_index <= 0;
                cell_accumulator <= 0;
                cell_response_reg <= 0;
                top0_valid <= '0';
                top1_valid <= '0';
                top0_x <= GATE_X;
                top1_x <= GATE_X;
                top0_y <= 0;
                top1_y <= 0;
                top0_score <= 0;
                top1_score <= 0;
            else
                case state is
                    when collect_image =>
                        if s_axis_tvalid = '1' then
                            if write_count = IMAGE_PIXELS - 1 then
                                write_count <= IMAGE_PIXELS;
                                radius_index <= 0;
                                ring_center_y <= 22;
                                direction_index <= 0;
                                ring_phase <= 0;
                                ring_score_acc <= 0;
                                ring_support_acc <= 0;
                                state <= ring_sample_prepare;
                            else
                                write_count <= write_count + 1;
                                write_address <= write_address + 1;
                            end if;
                        end if;

                    when ring_sample_prepare =>
                        radius_value := 18 + 2 * integer(radius_index);
                        if ring_phase = 0 then
                            sample_radius := radius_value - RADIAL_DELTA;
                        elsif ring_phase = 1 then
                            sample_radius := radius_value;
                        else
                            sample_radius := radius_value + RADIAL_DELTA;
                        end if;
                        sample_radius_reg <= natural(sample_radius);
                        sample_direction_x_reg <=
                            direction_x_q10(direction_index);
                        sample_direction_y_reg <=
                            direction_y_q10(direction_index);
                        state <= ring_offset_compute;

                    when ring_offset_compute =>
                        sample_offset_x_reg <= scale_q10(
                            integer(sample_radius_reg),
                            sample_direction_x_reg
                        );
                        sample_offset_y_reg <= scale_q10(
                            integer(sample_radius_reg),
                            sample_direction_y_reg
                        );
                        state <= ring_coordinate_compute;

                    when ring_coordinate_compute =>
                        sample_x := integer(GATE_X) + sample_offset_x_reg;
                        sample_y := integer(ring_center_y) + sample_offset_y_reg;
                        sample_x_reg <= natural(sample_x);
                        sample_y_reg <= natural(sample_y);
                        state <= ring_row_compute;

                    when ring_row_compute =>
                        sample_row_base_reg <= sample_y_reg * IMAGE_SIZE;
                        state <= ring_address_compute;

                    when ring_address_compute =>
                        read_address <= sample_row_base_reg + sample_x_reg;
                        state <= ring_issue;

                    when ring_issue =>
                        state <= ring_wait;

                    when ring_wait =>
                        state <= ring_consume;

                    when ring_consume =>
                        sample_value := to_integer(
                            unsigned(feature_read_data(15 downto 8))
                        );
                        if ring_phase = 0 then
                            ring_inner <= sample_value;
                            ring_phase <= 1;
                            state <= ring_sample_prepare;
                        elsif ring_phase = 1 then
                            ring_value <= sample_value;
                            ring_phase <= 2;
                            state <= ring_sample_prepare;
                        else
                            ring_outer <= sample_value;
                            state <= ring_contrast_compute;
                        end if;

                    when ring_contrast_compute =>
                            ring_contrast_reg <= ring_inner + ring_outer -
                                2 * ring_value;
                            state <= ring_accumulate;

                    when ring_accumulate =>
                            contrast_value := ring_contrast_reg;
                            score_after := ring_score_acc;
                            support_after := ring_support_acc;
                            if contrast_value > 0 then
                                score_after := score_after + contrast_value;
                            end if;
                            if contrast_value > SUPPORT_THRESHOLD then
                                support_after := support_after + 1;
                            end if;
                            if direction_index = DIRECTION_COUNT - 1 then
                                completed_ring_score <= natural(score_after);
                                completed_ring_support <= support_after;
                                state <= ring_finalize_score;
                            else
                                ring_score_acc <= score_after;
                                ring_support_acc <= support_after;
                                direction_index <= direction_index + 1;
                                ring_phase <= 0;
                                state <= ring_sample_prepare;
                            end if;

                    when ring_finalize_score =>
                        final_score := integer(completed_ring_score) +
                            SUPPORT_BONUS * integer(completed_ring_support);
                        final_ring_score_reg <= natural(final_score);
                        state <= ring_compare_best;

                    when ring_compare_best =>
                        if final_ring_score_reg > best_ring_score then
                            best_ring_score <= final_ring_score_reg;
                            best_ring_y <= ring_center_y;
                            best_ring_radius <= 18 + 2 * radius_index;
                        end if;
                        state <= ring_advance;

                    when ring_advance =>
                        radius_value := 18 + 2 * integer(radius_index);
                        margin_value := radius_value + RADIAL_DELTA;
                        if ring_center_y = IMAGE_SIZE - margin_value - 1 then
                            if radius_index = RADIUS_COUNT - 1 then
                                cell_x_scan <= CELL_X_MIN;
                                cell_y_scan <= CELL_Y_MIN;
                                cell_sample_index <= 0;
                                cell_accumulator <= 0;
                                state <= cell_sample_prepare;
                            else
                                radius_index <= radius_index + 1;
                                radius_value := radius_value + 2;
                                ring_center_y <= radius_value + RADIAL_DELTA;
                                direction_index <= 0;
                                ring_phase <= 0;
                                ring_score_acc <= 0;
                                ring_support_acc <= 0;
                                state <= ring_sample_prepare;
                            end if;
                        else
                            ring_center_y <= ring_center_y + 1;
                            direction_index <= 0;
                            ring_phase <= 0;
                            ring_score_acc <= 0;
                            ring_support_acc <= 0;
                            state <= ring_sample_prepare;
                        end if;

                    when cell_sample_prepare =>
                        sample_x := integer(cell_x_scan) +
                            cell_offset_x(cell_sample_index);
                        sample_y := integer(cell_y_scan) +
                            cell_offset_y(cell_sample_index);
                        sample_x_reg <= natural(sample_x);
                        sample_y_reg <= natural(sample_y);
                        state <= cell_row_compute;

                    when cell_row_compute =>
                        sample_row_base_reg <= sample_y_reg * IMAGE_SIZE;
                        state <= cell_address_compute;

                    when cell_address_compute =>
                        read_address <= sample_row_base_reg + sample_x_reg;
                        state <= cell_issue;

                    when cell_issue =>
                        state <= cell_wait;

                    when cell_wait =>
                        state <= cell_consume;

                    when cell_consume =>
                        sample_value := to_integer(
                            unsigned(feature_read_data(7 downto 0))
                        );
                        if cell_sample_index = 0 then
                            cell_accumulator <= 8 * sample_value;
                            cell_sample_index <= 1;
                            state <= cell_sample_prepare;
                        elsif cell_sample_index < 8 then
                            cell_accumulator <= cell_accumulator - sample_value;
                            cell_sample_index <= cell_sample_index + 1;
                            state <= cell_sample_prepare;
                        else
                            cell_response_reg <=
                                cell_accumulator - sample_value;
                            state <= cell_candidate_update;
                        end if;

                    when cell_candidate_update =>
                            response_value := cell_response_reg;
                            if cell_response_reg >= CELL_RESPONSE_THRESHOLD_X8 then
                                clipped_response := natural(cell_response_reg / 8);
                                if clipped_response > 255 then
                                    clipped_response := 255;
                                end if;
                                close_top0 := top0_valid = '1' and
                                    abs_difference(cell_y_scan, top0_y) <
                                    CELL_MIN_Y_DISTANCE;
                                close_top1 := top1_valid = '1' and
                                    abs_difference(cell_y_scan, top1_y) <
                                    CELL_MIN_Y_DISTANCE;
                                if close_top0 then
                                    if clipped_response > top0_score then
                                        top0_score <= clipped_response;
                                        top0_x <= cell_x_scan;
                                        top0_y <= cell_y_scan;
                                    end if;
                                elsif close_top1 then
                                    if clipped_response > top1_score then
                                        top1_score <= clipped_response;
                                        top1_x <= cell_x_scan;
                                        top1_y <= cell_y_scan;
                                    end if;
                                elsif top0_valid = '0' then
                                    top0_valid <= '1';
                                    top0_score <= clipped_response;
                                    top0_x <= cell_x_scan;
                                    top0_y <= cell_y_scan;
                                elsif clipped_response > top0_score then
                                    top1_valid <= '1';
                                    top1_score <= top0_score;
                                    top1_x <= top0_x;
                                    top1_y <= top0_y;
                                    top0_score <= clipped_response;
                                    top0_x <= cell_x_scan;
                                    top0_y <= cell_y_scan;
                                elsif top1_valid = '0' or
                                      clipped_response > top1_score then
                                    top1_valid <= '1';
                                    top1_score <= clipped_response;
                                    top1_x <= cell_x_scan;
                                    top1_y <= cell_y_scan;
                                end if;
                            end if;
                            state <= cell_scan_advance;

                    when cell_scan_advance =>
                            cell_sample_index <= 0;
                            cell_accumulator <= 0;
                            if cell_y_scan = CELL_Y_MAX then
                                cell_y_scan <= CELL_Y_MIN;
                                if cell_x_scan = CELL_X_MAX then
                                    state <= hold_result;
                                else
                                    cell_x_scan <= cell_x_scan + 1;
                                    state <= cell_sample_prepare;
                                end if;
                            else
                                cell_y_scan <= cell_y_scan + 1;
                                state <= cell_sample_prepare;
                            end if;

                    when hold_result =>
                        null;
                end case;
            end if;
        end if;
    end process;
end architecture rtl;
