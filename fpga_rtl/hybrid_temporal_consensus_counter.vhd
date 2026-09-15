library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity hybrid_temporal_consensus_counter is
    generic (
        HISTORY_DEPTH             : positive := 64;
        MINIMUM_DELAY_FRAMES      : natural := 1;
        MAXIMUM_DELAY_FRAMES      : natural := 60;
        NOMINAL_DELAY_FRAMES      : natural := 12;
        MAXIMUM_Y_DISTANCE        : natural := 29;
        DROPLET_REFRACTORY_FRAMES : natural := 7;
        CELL_REQUIRED_HITS        : natural := 2;
        CELL_MAXIMUM_Y_MOTION     : natural := 7;
        CELL_REFRACTORY_FRAMES    : natural := 7;
        QNN_DROPLET_THRESHOLD     : natural := 66;
        CLASSICAL_DROP_THRESHOLD  : natural := 863
    );
    port (
        clk     : in std_logic;
        reset_n : in std_logic;
        frame_start : in std_logic;

        qnn_frame_done      : in std_logic;
        qnn_droplet_present : in std_logic;
        qnn_droplet_y       : in std_logic_vector(6 downto 0);
        qnn_droplet_score   : in std_logic_vector(7 downto 0);
        qnn_cell_count      : in std_logic_vector(1 downto 0);
        qnn_cell0_y         : in std_logic_vector(6 downto 0);
        qnn_cell1_y         : in std_logic_vector(6 downto 0);

        classical_frame_done      : in std_logic;
        classical_droplet_present : in std_logic;
        classical_droplet_y       : in std_logic_vector(6 downto 0);
        classical_droplet_score   : in std_logic_vector(15 downto 0);
        classical_cell_count      : in std_logic_vector(1 downto 0);
        classical_cell0_y         : in std_logic_vector(6 downto 0);
        classical_cell1_y         : in std_logic_vector(6 downto 0);

        status_valid : out std_logic;
        status_flags : out std_logic_vector(7 downto 0);
        droplet_count : out std_logic_vector(15 downto 0);
        cell_count : out std_logic_vector(15 downto 0)
    );
end entity hybrid_temporal_consensus_counter;

architecture rtl of hybrid_temporal_consensus_counter is
    subtype frame_number_t is unsigned(15 downto 0);
    type control_state_t is (
        wait_for_frame,
        wait_for_inputs,
        prepare_events,
        qnn_cell_motion,
        qnn_cell_refractory,
        qnn_cell_commit,
        classical_cell_motion,
        classical_cell_refractory,
        classical_cell_commit,
        insert_qnn_events,
        select_classical_event,
        scan_history,
        measure_history,
        score_history,
        compare_history,
        finish_match,
        emit_status
    );
    type history_valid_array_t is array (0 to HISTORY_DEPTH - 1) of std_logic;
    type history_class_array_t is array (0 to HISTORY_DEPTH - 1) of
        natural range 0 to 1;
    type history_y_array_t is array (0 to HISTORY_DEPTH - 1) of
        natural range 0 to 127;
    type history_frame_array_t is array (0 to HISTORY_DEPTH - 1) of
        frame_number_t;

    type event_valid_array_t is array (0 to 2) of std_logic;
    type event_class_array_t is array (0 to 2) of natural range 0 to 1;
    type event_y_array_t is array (0 to 2) of natural range 0 to 127;
    type event_frame_array_t is array (0 to 2) of frame_number_t;

    type recent_valid_array_t is array (0 to 1) of std_logic;
    type recent_y_array_t is array (0 to 1) of natural range 0 to 127;
    type recent_frame_array_t is array (0 to 1) of frame_number_t;

    signal state : control_state_t := wait_for_frame;
    signal frame_sequence : frame_number_t := (others => '0');
    signal active_frame : frame_number_t := (others => '0');
    signal qnn_seen : std_logic := '0';
    signal classical_seen : std_logic := '0';

    signal qnn_drop_present_reg : std_logic := '0';
    signal qnn_drop_y_reg : natural range 0 to 127 := 0;
    signal qnn_drop_score_reg : natural range 0 to 127 := 0;
    signal qnn_cells_reg : natural range 0 to 2 := 0;
    signal qnn_cell0_y_reg : natural range 0 to 127 := 0;
    signal qnn_cell1_y_reg : natural range 0 to 127 := 0;

    signal classical_drop_present_reg : std_logic := '0';
    signal classical_drop_y_reg : natural range 0 to 127 := 0;
    signal classical_drop_score_reg : natural range 0 to 65535 := 0;
    signal classical_cells_reg : natural range 0 to 2 := 0;
    signal classical_cell0_y_reg : natural range 0 to 127 := 0;
    signal classical_cell1_y_reg : natural range 0 to 127 := 0;

    signal qnn_drop_score_prev2 : natural range 0 to 127 := 0;
    signal qnn_drop_score_prev1 : natural range 0 to 127 := 0;
    signal qnn_drop_y_prev1 : natural range 0 to 127 := 0;
    signal qnn_last_drop_valid : std_logic := '0';
    signal qnn_last_drop_frame : frame_number_t := (others => '0');
    signal classical_drop_score_prev2 : natural range 0 to 65535 := 0;
    signal classical_drop_score_prev1 : natural range 0 to 65535 := 0;
    signal classical_drop_y_prev1 : natural range 0 to 127 := 0;
    signal classical_last_drop_valid : std_logic := '0';
    signal classical_last_drop_frame : frame_number_t := (others => '0');

    signal qnn_previous_cell_count : natural range 0 to 2 := 0;
    signal qnn_previous_cell0_y : natural range 0 to 127 := 0;
    signal qnn_previous_cell1_y : natural range 0 to 127 := 0;
    signal classical_previous_cell_count : natural range 0 to 2 := 0;
    signal classical_previous_cell0_y : natural range 0 to 127 := 0;
    signal classical_previous_cell1_y : natural range 0 to 127 := 0;

    signal qnn_recent_valid : recent_valid_array_t := (others => '0');
    signal qnn_recent_y : recent_y_array_t := (others => 0);
    signal qnn_recent_frame : recent_frame_array_t := (others => (others => '0'));
    signal qnn_recent_pointer : natural range 0 to 1 := 0;
    signal classical_recent_valid : recent_valid_array_t := (others => '0');
    signal classical_recent_y : recent_y_array_t := (others => 0);
    signal classical_recent_frame : recent_frame_array_t :=
        (others => (others => '0'));
    signal classical_recent_pointer : natural range 0 to 1 := 0;

    signal history_valid : history_valid_array_t := (others => '0');
    signal history_class : history_class_array_t := (others => 0);
    signal history_y : history_y_array_t := (others => 0);
    signal history_frame : history_frame_array_t :=
        (others => (others => '0'));
    signal history_write_pointer : natural range 0 to HISTORY_DEPTH - 1 := 0;

    signal qnn_event_valid : event_valid_array_t := (others => '0');
    signal qnn_event_class : event_class_array_t := (others => 0);
    signal qnn_event_y : event_y_array_t := (others => 0);
    signal qnn_event_frame : event_frame_array_t :=
        (others => (others => '0'));
    signal classical_event_valid : event_valid_array_t := (others => '0');
    signal classical_event_class : event_class_array_t := (others => 0);
    signal classical_event_y : event_y_array_t := (others => 0);
    signal classical_event_frame : event_frame_array_t :=
        (others => (others => '0'));

    signal cell_candidate_index : natural range 0 to 1 := 0;
    signal qnn_cell_event_slot : natural range 0 to 1 := 0;
    signal classical_cell_event_slot : natural range 0 to 1 := 0;
    signal pending_cell_y : natural range 0 to 127 := 0;
    signal pending_cell_available : std_logic := '0';
    signal pending_cell_motion_match : std_logic := '0';
    signal pending_cell_is_event : std_logic := '0';

    signal qnn_insert_index : natural range 0 to 3 := 0;
    signal classical_match_index : natural range 0 to 3 := 0;
    signal history_scan_index : natural range 0 to HISTORY_DEPTH - 1 := 0;
    signal scan_entry_valid : std_logic := '0';
    signal scan_entry_class : natural range 0 to 1 := 0;
    signal scan_entry_y : natural range 0 to 127 := 0;
    signal scan_entry_frame : frame_number_t := (others => '0');
    signal scan_entry_index : natural range 0 to HISTORY_DEPTH - 1 := 0;
    signal scan_class_match : std_logic := '0';
    signal scan_delay : natural range 0 to 65535 := 0;
    signal scan_y_distance : natural range 0 to 127 := 0;
    signal scan_candidate_eligible : std_logic := '0';
    signal scan_candidate_cost : natural range 0 to 65535 := 0;
    signal best_match_valid : std_logic := '0';
    signal best_match_index : natural range 0 to HISTORY_DEPTH - 1 := 0;
    signal best_match_cost : natural range 0 to 65535 := 65535;

    signal status_valid_reg : std_logic := '0';
    signal status_flags_reg : std_logic_vector(7 downto 0) := (others => '0');
    signal droplet_count_reg : unsigned(15 downto 0) := (others => '0');
    signal cell_count_reg : unsigned(15 downto 0) := (others => '0');

    function absolute_difference(left_value : natural; right_value : natural)
        return natural is
    begin
        if left_value >= right_value then
            return left_value - right_value;
        end if;
        return right_value - left_value;
    end function;

    function matches_previous(
        candidate_y : natural;
        previous_count : natural;
        previous_y0 : natural;
        previous_y1 : natural
    ) return boolean is
    begin
        if previous_count >= 1 and
           absolute_difference(candidate_y, previous_y0) <=
           CELL_MAXIMUM_Y_MOTION then
            return true;
        end if;
        return previous_count >= 2 and
               absolute_difference(candidate_y, previous_y1) <=
               CELL_MAXIMUM_Y_MOTION;
    end function;

    function recently_emitted(
        candidate_y : natural;
        candidate_frame : frame_number_t;
        valid_values : recent_valid_array_t;
        y_values : recent_y_array_t;
        frame_values : recent_frame_array_t
    ) return boolean is
        variable age : natural;
    begin
        for index in 0 to 1 loop
            if valid_values(index) = '1' and
               absolute_difference(candidate_y, y_values(index)) <=
               CELL_MAXIMUM_Y_MOTION then
                age := to_integer(candidate_frame - frame_values(index));
                if age < CELL_REFRACTORY_FRAMES then
                    return true;
                end if;
            end if;
        end loop;
        return false;
    end function;

    procedure record_recent(
        constant candidate_y : in natural;
        constant candidate_frame : in frame_number_t;
        variable valid_values : inout recent_valid_array_t;
        variable y_values : inout recent_y_array_t;
        variable frame_values : inout recent_frame_array_t;
        variable pointer_value : inout natural
    ) is
    begin
        valid_values(pointer_value) := '1';
        y_values(pointer_value) := candidate_y;
        frame_values(pointer_value) := candidate_frame;
        if pointer_value = 0 then
            pointer_value := 1;
        else
            pointer_value := 0;
        end if;
    end procedure;
begin
    assert CELL_REQUIRED_HITS = 2
        report "This compact temporal implementation is calibrated for two hits"
        severity failure;
    assert MINIMUM_DELAY_FRAMES <= NOMINAL_DELAY_FRAMES and
           NOMINAL_DELAY_FRAMES <= MAXIMUM_DELAY_FRAMES
        report "Nominal delay must lie inside the fusion window"
        severity failure;

    status_valid <= status_valid_reg;
    status_flags <= status_flags_reg;
    droplet_count <= std_logic_vector(droplet_count_reg);
    cell_count <= std_logic_vector(cell_count_reg);

    process (clk)
        variable q_valid_v : event_valid_array_t;
        variable q_class_v : event_class_array_t;
        variable q_y_v : event_y_array_t;
        variable q_frame_v : event_frame_array_t;
        variable c_valid_v : event_valid_array_t;
        variable c_class_v : event_class_array_t;
        variable c_y_v : event_y_array_t;
        variable c_frame_v : event_frame_array_t;
        variable flags_v : std_logic_vector(7 downto 0);
        variable current_qnn_drop_score : natural;
        variable peak_frame : frame_number_t;
        variable drop_age : natural;
        variable q_recent_valid_v : recent_valid_array_t;
        variable q_recent_y_v : recent_y_array_t;
        variable q_recent_frame_v : recent_frame_array_t;
        variable q_recent_pointer_v : natural range 0 to 1;
        variable c_recent_valid_v : recent_valid_array_t;
        variable c_recent_y_v : recent_y_array_t;
        variable c_recent_frame_v : recent_frame_array_t;
        variable c_recent_pointer_v : natural range 0 to 1;
        variable candidate_y : natural;
        variable candidate_is_event : boolean;
        variable event_slot : natural range 0 to 1;
        variable delay_value : natural;
        variable y_distance : natural;
        variable time_distance : natural;
        variable candidate_cost : natural;
    begin
        if rising_edge(clk) then
            status_valid_reg <= '0';
            if reset_n = '0' then
                state <= wait_for_frame;
                frame_sequence <= (others => '0');
                active_frame <= (others => '0');
                qnn_seen <= '0';
                classical_seen <= '0';
                qnn_drop_present_reg <= '0';
                qnn_drop_y_reg <= 0;
                qnn_drop_score_reg <= 0;
                qnn_cells_reg <= 0;
                qnn_cell0_y_reg <= 0;
                qnn_cell1_y_reg <= 0;
                classical_drop_present_reg <= '0';
                classical_drop_y_reg <= 0;
                classical_drop_score_reg <= 0;
                classical_cells_reg <= 0;
                classical_cell0_y_reg <= 0;
                classical_cell1_y_reg <= 0;
                qnn_drop_score_prev2 <= 0;
                qnn_drop_score_prev1 <= 0;
                qnn_drop_y_prev1 <= 0;
                qnn_last_drop_valid <= '0';
                qnn_last_drop_frame <= (others => '0');
                classical_drop_score_prev2 <= 0;
                classical_drop_score_prev1 <= 0;
                classical_drop_y_prev1 <= 0;
                classical_last_drop_valid <= '0';
                classical_last_drop_frame <= (others => '0');
                qnn_previous_cell_count <= 0;
                qnn_previous_cell0_y <= 0;
                qnn_previous_cell1_y <= 0;
                classical_previous_cell_count <= 0;
                classical_previous_cell0_y <= 0;
                classical_previous_cell1_y <= 0;
                qnn_recent_valid <= (others => '0');
                qnn_recent_y <= (others => 0);
                qnn_recent_frame <= (others => (others => '0'));
                qnn_recent_pointer <= 0;
                classical_recent_valid <= (others => '0');
                classical_recent_y <= (others => 0);
                classical_recent_frame <= (others => (others => '0'));
                classical_recent_pointer <= 0;
                history_valid <= (others => '0');
                history_class <= (others => 0);
                history_y <= (others => 0);
                history_frame <= (others => (others => '0'));
                history_write_pointer <= 0;
                qnn_event_valid <= (others => '0');
                qnn_event_class <= (others => 0);
                qnn_event_y <= (others => 0);
                qnn_event_frame <= (others => (others => '0'));
                classical_event_valid <= (others => '0');
                classical_event_class <= (others => 0);
                classical_event_y <= (others => 0);
                classical_event_frame <= (others => (others => '0'));
                cell_candidate_index <= 0;
                qnn_cell_event_slot <= 0;
                classical_cell_event_slot <= 0;
                pending_cell_y <= 0;
                pending_cell_available <= '0';
                pending_cell_motion_match <= '0';
                pending_cell_is_event <= '0';
                qnn_insert_index <= 0;
                classical_match_index <= 0;
                history_scan_index <= 0;
                scan_entry_valid <= '0';
                scan_entry_class <= 0;
                scan_entry_y <= 0;
                scan_entry_frame <= (others => '0');
                scan_entry_index <= 0;
                scan_class_match <= '0';
                scan_delay <= 0;
                scan_y_distance <= 0;
                scan_candidate_eligible <= '0';
                scan_candidate_cost <= 0;
                best_match_valid <= '0';
                best_match_index <= 0;
                best_match_cost <= 65535;
                status_flags_reg <= (others => '0');
                droplet_count_reg <= (others => '0');
                cell_count_reg <= (others => '0');
            elsif frame_start = '1' then
                active_frame <= frame_sequence;
                frame_sequence <= frame_sequence + 1;
                qnn_seen <= '0';
                classical_seen <= '0';
                state <= wait_for_inputs;
            else
                case state is
                    when wait_for_frame =>
                        null;

                    when wait_for_inputs =>
                        if qnn_frame_done = '1' then
                            qnn_seen <= '1';
                            qnn_drop_present_reg <= qnn_droplet_present;
                            qnn_drop_y_reg <= to_integer(unsigned(qnn_droplet_y));
                            qnn_drop_score_reg <=
                                to_integer(unsigned(qnn_droplet_score));
                            qnn_cells_reg <= to_integer(unsigned(qnn_cell_count));
                            qnn_cell0_y_reg <= to_integer(unsigned(qnn_cell0_y));
                            qnn_cell1_y_reg <= to_integer(unsigned(qnn_cell1_y));
                        end if;
                        if classical_frame_done = '1' and classical_seen = '0' then
                            classical_seen <= '1';
                            classical_drop_present_reg <=
                                classical_droplet_present;
                            classical_drop_y_reg <=
                                to_integer(unsigned(classical_droplet_y));
                            classical_drop_score_reg <=
                                to_integer(unsigned(classical_droplet_score));
                            classical_cells_reg <=
                                to_integer(unsigned(classical_cell_count));
                            classical_cell0_y_reg <=
                                to_integer(unsigned(classical_cell0_y));
                            classical_cell1_y_reg <=
                                to_integer(unsigned(classical_cell1_y));
                        end if;
                        if (qnn_seen = '1' or qnn_frame_done = '1') and
                           (classical_seen = '1' or classical_frame_done = '1') then
                            state <= prepare_events;
                        end if;

                    when prepare_events =>
                        q_valid_v := (others => '0');
                        q_class_v := (others => 0);
                        q_y_v := (others => 0);
                        q_frame_v := (others => active_frame);
                        c_valid_v := (others => '0');
                        c_class_v := (others => 0);
                        c_y_v := (others => 0);
                        c_frame_v := (others => active_frame);
                        flags_v := "11000000";

                        q_recent_valid_v := qnn_recent_valid;
                        q_recent_y_v := qnn_recent_y;
                        q_recent_frame_v := qnn_recent_frame;
                        q_recent_pointer_v := qnn_recent_pointer;
                        c_recent_valid_v := classical_recent_valid;
                        c_recent_y_v := classical_recent_y;
                        c_recent_frame_v := classical_recent_frame;
                        c_recent_pointer_v := classical_recent_pointer;

                        if qnn_drop_present_reg = '1' then
                            current_qnn_drop_score := qnn_drop_score_reg;
                        else
                            current_qnn_drop_score := 0;
                        end if;
                        peak_frame := active_frame - 1;
                        if qnn_drop_score_prev1 >= QNN_DROPLET_THRESHOLD and
                           qnn_drop_score_prev1 >= qnn_drop_score_prev2 and
                           qnn_drop_score_prev1 > current_qnn_drop_score then
                            if qnn_last_drop_valid = '0' then
                                candidate_is_event := true;
                            else
                                drop_age := to_integer(
                                    peak_frame - qnn_last_drop_frame
                                );
                                candidate_is_event :=
                                    drop_age >= DROPLET_REFRACTORY_FRAMES;
                            end if;
                            if candidate_is_event then
                                q_valid_v(2) := '1';
                                q_class_v(2) := 1;
                                q_y_v(2) := qnn_drop_y_prev1;
                                q_frame_v(2) := peak_frame;
                                flags_v(0) := '1';
                                qnn_last_drop_valid <= '1';
                                qnn_last_drop_frame <= peak_frame;
                            end if;
                        end if;
                        qnn_drop_score_prev2 <= qnn_drop_score_prev1;
                        qnn_drop_score_prev1 <= current_qnn_drop_score;
                        qnn_drop_y_prev1 <= qnn_drop_y_reg;

                        peak_frame := active_frame - 1;
                        if classical_drop_score_prev1 >=
                           CLASSICAL_DROP_THRESHOLD and
                           classical_drop_score_prev1 >=
                           classical_drop_score_prev2 and
                           classical_drop_score_prev1 >
                           classical_drop_score_reg then
                            if classical_last_drop_valid = '0' then
                                candidate_is_event := true;
                            else
                                drop_age := to_integer(
                                    peak_frame - classical_last_drop_frame
                                );
                                candidate_is_event :=
                                    drop_age >= DROPLET_REFRACTORY_FRAMES;
                            end if;
                            if candidate_is_event then
                                c_valid_v(2) := '1';
                                c_class_v(2) := 1;
                                c_y_v(2) := classical_drop_y_prev1;
                                c_frame_v(2) := peak_frame;
                                flags_v(2) := '1';
                                classical_last_drop_valid <= '1';
                                classical_last_drop_frame <= peak_frame;
                            end if;
                        end if;
                        classical_drop_score_prev2 <=
                            classical_drop_score_prev1;
                        classical_drop_score_prev1 <=
                            classical_drop_score_reg;
                        classical_drop_y_prev1 <= classical_drop_y_reg;

                        qnn_event_valid <= q_valid_v;
                        qnn_event_class <= q_class_v;
                        qnn_event_y <= q_y_v;
                        qnn_event_frame <= q_frame_v;
                        classical_event_valid <= c_valid_v;
                        classical_event_class <= c_class_v;
                        classical_event_y <= c_y_v;
                        classical_event_frame <= c_frame_v;
                        status_flags_reg <= flags_v;
                        cell_candidate_index <= 0;
                        qnn_cell_event_slot <= 0;
                        classical_cell_event_slot <= 0;
                        state <= qnn_cell_motion;

                    when qnn_cell_motion =>
                        if cell_candidate_index = 0 then
                            candidate_y := qnn_cell0_y_reg;
                        else
                            candidate_y := qnn_cell1_y_reg;
                        end if;
                        pending_cell_y <= candidate_y;
                        if qnn_cells_reg > cell_candidate_index then
                            pending_cell_available <= '1';
                        else
                            pending_cell_available <= '0';
                        end if;
                        if matches_previous(
                            candidate_y,
                            qnn_previous_cell_count,
                            qnn_previous_cell0_y,
                            qnn_previous_cell1_y
                        ) then
                            pending_cell_motion_match <= '1';
                        else
                            pending_cell_motion_match <= '0';
                        end if;
                        state <= qnn_cell_refractory;

                    when qnn_cell_refractory =>
                        if pending_cell_available = '1' and
                           pending_cell_motion_match = '1' and
                           not recently_emitted(
                               pending_cell_y,
                               active_frame,
                               qnn_recent_valid,
                               qnn_recent_y,
                               qnn_recent_frame
                           ) then
                            pending_cell_is_event <= '1';
                        else
                            pending_cell_is_event <= '0';
                        end if;
                        state <= qnn_cell_commit;

                    when qnn_cell_commit =>
                        if pending_cell_is_event = '1' then
                            qnn_event_valid(qnn_cell_event_slot) <= '1';
                            qnn_event_class(qnn_cell_event_slot) <= 0;
                            qnn_event_y(qnn_cell_event_slot) <= pending_cell_y;
                            qnn_event_frame(qnn_cell_event_slot) <= active_frame;
                            status_flags_reg(1) <= '1';
                            qnn_recent_valid(qnn_recent_pointer) <= '1';
                            qnn_recent_y(qnn_recent_pointer) <= pending_cell_y;
                            qnn_recent_frame(qnn_recent_pointer) <= active_frame;
                            if qnn_recent_pointer = 0 then
                                qnn_recent_pointer <= 1;
                            else
                                qnn_recent_pointer <= 0;
                            end if;
                            if qnn_cell_event_slot = 0 then
                                qnn_cell_event_slot <= 1;
                            end if;
                        end if;
                        if cell_candidate_index = 0 then
                            cell_candidate_index <= 1;
                            state <= qnn_cell_motion;
                        else
                            qnn_previous_cell_count <= qnn_cells_reg;
                            qnn_previous_cell0_y <= qnn_cell0_y_reg;
                            qnn_previous_cell1_y <= qnn_cell1_y_reg;
                            cell_candidate_index <= 0;
                            state <= classical_cell_motion;
                        end if;

                    when classical_cell_motion =>
                        if cell_candidate_index = 0 then
                            candidate_y := classical_cell0_y_reg;
                        else
                            candidate_y := classical_cell1_y_reg;
                        end if;
                        pending_cell_y <= candidate_y;
                        if classical_cells_reg > cell_candidate_index then
                            pending_cell_available <= '1';
                        else
                            pending_cell_available <= '0';
                        end if;
                        if matches_previous(
                            candidate_y,
                            classical_previous_cell_count,
                            classical_previous_cell0_y,
                            classical_previous_cell1_y
                        ) then
                            pending_cell_motion_match <= '1';
                        else
                            pending_cell_motion_match <= '0';
                        end if;
                        state <= classical_cell_refractory;

                    when classical_cell_refractory =>
                        if pending_cell_available = '1' and
                           pending_cell_motion_match = '1' and
                           not recently_emitted(
                               pending_cell_y,
                               active_frame,
                               classical_recent_valid,
                               classical_recent_y,
                               classical_recent_frame
                           ) then
                            pending_cell_is_event <= '1';
                        else
                            pending_cell_is_event <= '0';
                        end if;
                        state <= classical_cell_commit;

                    when classical_cell_commit =>
                        if pending_cell_is_event = '1' then
                            classical_event_valid(classical_cell_event_slot) <= '1';
                            classical_event_class(classical_cell_event_slot) <= 0;
                            classical_event_y(classical_cell_event_slot) <=
                                pending_cell_y;
                            classical_event_frame(classical_cell_event_slot) <=
                                active_frame;
                            status_flags_reg(3) <= '1';
                            classical_recent_valid(classical_recent_pointer) <= '1';
                            classical_recent_y(classical_recent_pointer) <=
                                pending_cell_y;
                            classical_recent_frame(classical_recent_pointer) <=
                                active_frame;
                            if classical_recent_pointer = 0 then
                                classical_recent_pointer <= 1;
                            else
                                classical_recent_pointer <= 0;
                            end if;
                            if classical_cell_event_slot = 0 then
                                classical_cell_event_slot <= 1;
                            end if;
                        end if;
                        if cell_candidate_index = 0 then
                            cell_candidate_index <= 1;
                            state <= classical_cell_motion;
                        else
                            classical_previous_cell_count <= classical_cells_reg;
                            classical_previous_cell0_y <= classical_cell0_y_reg;
                            classical_previous_cell1_y <= classical_cell1_y_reg;
                            qnn_insert_index <= 0;
                            state <= insert_qnn_events;
                        end if;

                    when insert_qnn_events =>
                        if qnn_insert_index <= 2 then
                            if qnn_event_valid(qnn_insert_index) = '1' then
                                history_valid(history_write_pointer) <= '1';
                                history_class(history_write_pointer) <=
                                    qnn_event_class(qnn_insert_index);
                                history_y(history_write_pointer) <=
                                    qnn_event_y(qnn_insert_index);
                                history_frame(history_write_pointer) <=
                                    qnn_event_frame(qnn_insert_index);
                                if history_write_pointer = HISTORY_DEPTH - 1 then
                                    history_write_pointer <= 0;
                                else
                                    history_write_pointer <=
                                        history_write_pointer + 1;
                                end if;
                            end if;
                            qnn_insert_index <= qnn_insert_index + 1;
                        else
                            classical_match_index <= 0;
                            state <= select_classical_event;
                        end if;

                    when select_classical_event =>
                        if classical_match_index > 2 then
                            state <= emit_status;
                        elsif classical_event_valid(classical_match_index) = '0' then
                            classical_match_index <= classical_match_index + 1;
                        else
                            history_scan_index <= 0;
                            best_match_valid <= '0';
                            best_match_index <= 0;
                            best_match_cost <= 65535;
                            state <= scan_history;
                        end if;

                    when scan_history =>
                        scan_entry_valid <= history_valid(history_scan_index);
                        scan_entry_class <= history_class(history_scan_index);
                        scan_entry_y <= history_y(history_scan_index);
                        scan_entry_frame <= history_frame(history_scan_index);
                        scan_entry_index <= history_scan_index;
                        state <= measure_history;

                    when measure_history =>
                        if scan_entry_valid = '1' and
                           scan_entry_class =
                           classical_event_class(classical_match_index) then
                            scan_class_match <= '1';
                        else
                            scan_class_match <= '0';
                        end if;
                        delay_value := to_integer(
                            classical_event_frame(classical_match_index) -
                            scan_entry_frame
                        );
                        y_distance := absolute_difference(
                            classical_event_y(classical_match_index),
                            scan_entry_y
                        );
                        scan_delay <= delay_value;
                        scan_y_distance <= y_distance;
                        state <= score_history;

                    when score_history =>
                        if scan_class_match = '1' and
                           scan_delay >= MINIMUM_DELAY_FRAMES and
                           scan_delay <= MAXIMUM_DELAY_FRAMES and
                           scan_y_distance <= MAXIMUM_Y_DISTANCE then
                            time_distance := absolute_difference(
                                scan_delay, NOMINAL_DELAY_FRAMES
                            );
                            scan_candidate_cost <=
                                2 * time_distance + scan_y_distance;
                            scan_candidate_eligible <= '1';
                        else
                            scan_candidate_cost <= 0;
                            scan_candidate_eligible <= '0';
                        end if;
                        state <= compare_history;

                    when compare_history =>
                        if scan_candidate_eligible = '1' and
                           (best_match_valid = '0' or
                            scan_candidate_cost < best_match_cost) then
                            best_match_valid <= '1';
                            best_match_index <= scan_entry_index;
                            best_match_cost <= scan_candidate_cost;
                        end if;
                        if scan_entry_index = HISTORY_DEPTH - 1 then
                            state <= finish_match;
                        else
                            history_scan_index <= scan_entry_index + 1;
                            state <= scan_history;
                        end if;

                    when finish_match =>
                        if best_match_valid = '1' then
                            history_valid(best_match_index) <= '0';
                            if classical_event_class(classical_match_index) = 0 then
                                cell_count_reg <= cell_count_reg + 1;
                                status_flags_reg(5) <= '1';
                            else
                                droplet_count_reg <= droplet_count_reg + 1;
                                status_flags_reg(4) <= '1';
                            end if;
                        end if;
                        classical_match_index <= classical_match_index + 1;
                        state <= select_classical_event;

                    when emit_status =>
                        status_valid_reg <= '1';
                        state <= wait_for_frame;
                end case;
            end if;
        end if;
    end process;
end architecture rtl;
