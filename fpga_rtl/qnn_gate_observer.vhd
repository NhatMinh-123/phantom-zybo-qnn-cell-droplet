library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity qnn_gate_observer is
    generic (
        GRID_WIDTH          : positive := 24;
        GRID_HEIGHT         : positive := 24;
        GRID_STRIDE         : positive := 4;
        GRID_CENTER_OFFSET  : natural := 2;
        OUTPUT_CHANNELS     : positive := 15;
        CELL_OBJECT_CODE    : integer := 105;
        DROPLET_OBJECT_CODE : integer := 66;
        GATE_COLUMN_MIN     : natural := 11;
        GATE_COLUMN_MAX     : natural := 14;
        MINIMUM_Y_DISTANCE  : natural := 6
    );
    port (
        clk     : in std_logic;
        reset_n : in std_logic;

        stream_tdata  : in std_logic_vector(7 downto 0);
        stream_tvalid : in std_logic;
        stream_tready : in std_logic;

        frame_done       : out std_logic;
        droplet_present  : out std_logic;
        droplet_y        : out std_logic_vector(6 downto 0);
        droplet_score    : out std_logic_vector(7 downto 0);
        cell_count       : out std_logic_vector(1 downto 0);
        cell0_y          : out std_logic_vector(6 downto 0);
        cell0_score      : out std_logic_vector(7 downto 0);
        cell1_y          : out std_logic_vector(6 downto 0);
        cell1_score      : out std_logic_vector(7 downto 0)
    );
end entity qnn_gate_observer;

architecture rtl of qnn_gate_observer is
    type slot_values_t is array (0 to 4) of std_logic_vector(7 downto 0);
    signal slot_values : slot_values_t := (others => (others => '0'));
    signal field_index : natural range 0 to 4 := 0;
    signal slot_index : natural range 0 to 2 := 0;
    signal grid_x_reg : natural range 0 to GRID_WIDTH - 1 := 0;
    signal grid_y_reg : natural range 0 to GRID_HEIGHT - 1 := 0;

    signal candidate_pending_valid : std_logic := '0';
    signal candidate_pending_qualified : std_logic := '0';
    signal candidate_pending_slot : natural range 0 to 2 := 0;
    signal candidate_pending_y : natural range 0 to 127 := 0;
    signal candidate_pending_score : natural range 0 to 127 := 0;
    signal candidate_pending_last : std_logic := '0';

    signal droplet_valid_reg : std_logic := '0';
    signal droplet_y_reg : natural range 0 to 127 := 0;
    signal droplet_score_reg : natural range 0 to 127 := 0;
    signal cell0_valid_reg : std_logic := '0';
    signal cell1_valid_reg : std_logic := '0';
    signal cell0_y_reg : natural range 0 to 127 := 0;
    signal cell1_y_reg : natural range 0 to 127 := 0;
    signal cell0_score_reg : natural range 0 to 127 := 0;
    signal cell1_score_reg : natural range 0 to 127 := 0;
    signal frame_done_reg : std_logic := '0';

    function absolute_difference(left_value : natural; right_value : natural)
        return natural is
    begin
        if left_value >= right_value then
            return left_value - right_value;
        end if;
        return right_value - left_value;
    end function;
begin
    assert OUTPUT_CHANNELS = 15
        report "QNN gate observer expects three five-value slots"
        severity failure;
    assert GATE_COLUMN_MIN <= GATE_COLUMN_MAX and
           GATE_COLUMN_MAX < GRID_WIDTH
        report "QNN gate columns are outside the detector grid"
        severity failure;

    frame_done <= frame_done_reg;
    droplet_present <= droplet_valid_reg;
    droplet_y <= std_logic_vector(to_unsigned(droplet_y_reg, 7));
    droplet_score <= std_logic_vector(to_unsigned(droplet_score_reg, 8));
    cell_count <= "10" when cell1_valid_reg = '1' else
                  "01" when cell0_valid_reg = '1' else "00";
    cell0_y <= std_logic_vector(to_unsigned(cell0_y_reg, 7));
    cell0_score <= std_logic_vector(to_unsigned(cell0_score_reg, 8));
    cell1_y <= std_logic_vector(to_unsigned(cell1_y_reg, 7));
    cell1_score <= std_logic_vector(to_unsigned(cell1_score_reg, 8));

    process (clk)
        variable candidate_y : natural range 0 to 127;
        variable object_code : integer range -128 to 127;
        variable threshold_code : integer range -128 to 127;
    begin
        if rising_edge(clk) then
            frame_done_reg <= '0';
            if reset_n = '0' then
                slot_values <= (others => (others => '0'));
                field_index <= 0;
                slot_index <= 0;
                grid_x_reg <= 0;
                grid_y_reg <= 0;
                candidate_pending_valid <= '0';
                candidate_pending_qualified <= '0';
                candidate_pending_slot <= 0;
                candidate_pending_y <= 0;
                candidate_pending_score <= 0;
                candidate_pending_last <= '0';
                droplet_valid_reg <= '0';
                droplet_y_reg <= 0;
                droplet_score_reg <= 0;
                cell0_valid_reg <= '0';
                cell1_valid_reg <= '0';
                cell0_y_reg <= 0;
                cell1_y_reg <= 0;
                cell0_score_reg <= 0;
                cell1_score_reg <= 0;
            else
                if candidate_pending_valid = '1' then
                    if candidate_pending_qualified = '1' then
                        if candidate_pending_slot = 2 then
                            if droplet_valid_reg = '0' or
                               candidate_pending_score > droplet_score_reg then
                                droplet_valid_reg <= '1';
                                droplet_y_reg <= candidate_pending_y;
                                droplet_score_reg <= candidate_pending_score;
                            end if;
                        elsif cell0_valid_reg = '0' then
                            cell0_valid_reg <= '1';
                            cell0_y_reg <= candidate_pending_y;
                            cell0_score_reg <= candidate_pending_score;
                        elsif absolute_difference(
                                  candidate_pending_y, cell0_y_reg
                              ) < MINIMUM_Y_DISTANCE then
                            if candidate_pending_score > cell0_score_reg then
                                cell0_y_reg <= candidate_pending_y;
                                cell0_score_reg <= candidate_pending_score;
                            end if;
                        elsif cell1_valid_reg = '0' then
                            cell1_valid_reg <= '1';
                            cell1_y_reg <= candidate_pending_y;
                            cell1_score_reg <= candidate_pending_score;
                        elsif absolute_difference(
                                  candidate_pending_y, cell1_y_reg
                              ) < MINIMUM_Y_DISTANCE then
                            if candidate_pending_score > cell1_score_reg then
                                cell1_y_reg <= candidate_pending_y;
                                cell1_score_reg <= candidate_pending_score;
                            end if;
                        elsif candidate_pending_score > cell1_score_reg then
                            cell1_y_reg <= candidate_pending_y;
                            cell1_score_reg <= candidate_pending_score;
                        end if;
                    end if;
                    if candidate_pending_last = '1' then
                        frame_done_reg <= '1';
                    end if;
                    candidate_pending_valid <= '0';
                end if;

                if stream_tvalid = '1' and stream_tready = '1' then
                if field_index = 0 and slot_index = 0 and
                   grid_x_reg = 0 and grid_y_reg = 0 then
                    droplet_valid_reg <= '0';
                    droplet_y_reg <= 0;
                    droplet_score_reg <= 0;
                    cell0_valid_reg <= '0';
                    cell1_valid_reg <= '0';
                    cell0_y_reg <= 0;
                    cell1_y_reg <= 0;
                    cell0_score_reg <= 0;
                    cell1_score_reg <= 0;
                end if;

                slot_values(field_index) <= stream_tdata;

                if field_index = 4 then
                    candidate_y :=
                        grid_y_reg * GRID_STRIDE + GRID_CENTER_OFFSET;
                    object_code := to_integer(signed(slot_values(0)));
                    if slot_index < 2 then
                        threshold_code := CELL_OBJECT_CODE;
                    else
                        threshold_code := DROPLET_OBJECT_CODE;
                    end if;
                    candidate_pending_valid <= '1';
                    candidate_pending_slot <= slot_index;
                    candidate_pending_y <= candidate_y;
                    candidate_pending_last <= '1' when
                        slot_index = 2 and grid_x_reg = GRID_WIDTH - 1 and
                        grid_y_reg = GRID_HEIGHT - 1 else '0';
                    if grid_x_reg >= GATE_COLUMN_MIN and
                       grid_x_reg <= GATE_COLUMN_MAX and
                       object_code >= threshold_code then
                        candidate_pending_qualified <= '1';
                        candidate_pending_score <= natural(object_code);
                    else
                        candidate_pending_qualified <= '0';
                        candidate_pending_score <= 0;
                    end if;
                end if;

                if field_index = 4 then
                    field_index <= 0;
                    if slot_index = 2 then
                        slot_index <= 0;
                        if grid_x_reg = GRID_WIDTH - 1 then
                            grid_x_reg <= 0;
                            if grid_y_reg = GRID_HEIGHT - 1 then
                                grid_y_reg <= 0;
                            else
                                grid_y_reg <= grid_y_reg + 1;
                            end if;
                        else
                            grid_x_reg <= grid_x_reg + 1;
                        end if;
                    else
                        slot_index <= slot_index + 1;
                    end if;
                else
                    field_index <= field_index + 1;
                end if;
                end if;
            end if;
        end if;
    end process;
end architecture rtl;
