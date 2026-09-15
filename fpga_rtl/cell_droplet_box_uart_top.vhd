library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity cell_droplet_box_uart_top is
    generic (
        CLKS_PER_BIT          : positive := 104;
        CONFIDENCE_MIN_X1000 : natural  := 100
    );
    port (
        CLK12MHZ     : in  std_logic;
        uart_rxd_out : out std_logic;
        uart_txd_in  : in  std_logic;
        sw           : in  std_logic_vector(1 downto 0);
        led          : out std_logic_vector(3 downto 0)
    );
end entity cell_droplet_box_uart_top;

architecture rtl of cell_droplet_box_uart_top is
    type line_mode_t is (line_idle, line_frame, line_end, line_box);

    signal rx_valid : std_logic;
    signal rx_byte  : std_logic_vector(7 downto 0);
    signal tx_start : std_logic := '0';
    signal tx_byte  : std_logic_vector(7 downto 0) := (others => '0');
    signal tx_busy  : std_logic;

    signal line_mode     : line_mode_t := line_idle;
    signal current_value : unsigned(15 downto 0) := (others => '0');
    signal field_index   : integer range 0 to 5 := 0;
    signal digit_seen    : std_logic := '0';
    signal parse_error   : std_logic := '0';

    signal field0 : unsigned(15 downto 0) := (others => '0');
    signal field1 : unsigned(15 downto 0) := (others => '0');
    signal field2 : unsigned(15 downto 0) := (others => '0');
    signal field3 : unsigned(15 downto 0) := (others => '0');
    signal field4 : unsigned(15 downto 0) := (others => '0');

    signal frame_active       : std_logic := '0';
    signal cell_count         : unsigned(15 downto 0) := (others => '0');
    signal droplet_count      : unsigned(15 downto 0) := (others => '0');
    signal rejected_count     : unsigned(15 downto 0) := (others => '0');
    signal display_cell       : unsigned(15 downto 0) := (others => '0');
    signal display_droplet    : unsigned(15 downto 0) := (others => '0');
    signal display_rejected   : unsigned(15 downto 0) := (others => '0');

    signal ack_pending : std_logic := '0';
    signal ack_byte    : std_logic_vector(7 downto 0) := x"45";

    function is_digit(value : std_logic_vector(7 downto 0)) return boolean is
    begin
        return unsigned(value) >= to_unsigned(48, 8) and
               unsigned(value) <= to_unsigned(57, 8);
    end function;

    function is_separator(value : std_logic_vector(7 downto 0)) return boolean is
    begin
        return value = x"20" or value = x"09" or value = x"2C";
    end function;

    function is_newline(value : std_logic_vector(7 downto 0)) return boolean is
    begin
        return value = x"0A" or value = x"0D";
    end function;
begin
    rx_inst : entity work.uart_rx
        generic map (CLKS_PER_BIT => CLKS_PER_BIT)
        port map (
            clk       => CLK12MHZ,
            rx_serial => uart_txd_in,
            rx_valid  => rx_valid,
            rx_byte   => rx_byte
        );

    tx_inst : entity work.uart_tx
        generic map (CLKS_PER_BIT => CLKS_PER_BIT)
        port map (
            clk       => CLK12MHZ,
            tx_start  => tx_start,
            tx_byte   => tx_byte,
            tx_serial => uart_rxd_out,
            tx_busy   => tx_busy
        );

    process (CLK12MHZ)
        variable digit_value : unsigned(15 downto 0);
        variable new_value   : unsigned(15 downto 0);
        variable class_value : natural;
        variable x1_value    : natural;
        variable y1_value    : natural;
        variable x2_value    : natural;
        variable y2_value    : natural;
        variable conf_value  : natural;
        variable box_valid   : boolean;
    begin
        if rising_edge(CLK12MHZ) then
            tx_start <= '0';

            if ack_pending = '1' and tx_busy = '0' then
                tx_byte     <= ack_byte;
                tx_start    <= '1';
                ack_pending <= '0';
            end if;

            if rx_valid = '1' then
                if is_newline(rx_byte) then
                    case line_mode is
                        when line_frame =>
                            if parse_error = '0' and digit_seen = '0' then
                                frame_active     <= '1';
                                cell_count       <= (others => '0');
                                droplet_count    <= (others => '0');
                                rejected_count   <= (others => '0');
                                display_cell     <= (others => '0');
                                display_droplet  <= (others => '0');
                                display_rejected <= (others => '0');
                                ack_byte          <= x"46"; -- F: frame opened
                            else
                                ack_byte <= x"45"; -- E: protocol error
                            end if;
                            ack_pending <= '1';

                        when line_end =>
                            if parse_error = '0' and digit_seen = '0' and frame_active = '1' then
                                frame_active     <= '0';
                                display_cell     <= cell_count;
                                display_droplet  <= droplet_count;
                                display_rejected <= rejected_count;
                                ack_byte          <= x"44"; -- D: frame done
                            else
                                ack_byte <= x"45";
                            end if;
                            ack_pending <= '1';

                        when line_box =>
                            box_valid := false;
                            class_value := 0;
                            if parse_error = '0' and digit_seen = '1' and
                               field_index = 5 and frame_active = '1' then
                                class_value := to_integer(field0);
                                x1_value    := to_integer(field1);
                                y1_value    := to_integer(field2);
                                x2_value    := to_integer(field3);
                                y2_value    := to_integer(field4);
                                conf_value  := to_integer(current_value);
                                box_valid := class_value <= 1 and
                                             x1_value < x2_value and
                                             y1_value < y2_value and
                                             x2_value < 1280 and
                                             y2_value < 800 and
                                             conf_value >= CONFIDENCE_MIN_X1000;
                            end if;

                            if box_valid then
                                if class_value = 0 then
                                    cell_count <= cell_count + 1;
                                else
                                    droplet_count <= droplet_count + 1;
                                end if;
                                ack_byte <= x"42"; -- B: box accepted
                            else
                                if frame_active = '1' then
                                    rejected_count <= rejected_count + 1;
                                end if;
                                ack_byte <= x"52"; -- R: box rejected
                            end if;
                            ack_pending <= '1';

                        when line_idle =>
                            null;
                    end case;

                    line_mode     <= line_idle;
                    current_value <= (others => '0');
                    field_index   <= 0;
                    digit_seen    <= '0';
                    parse_error   <= '0';

                elsif line_mode = line_idle then
                    if rx_byte = x"46" or rx_byte = x"66" then
                        line_mode <= line_frame;
                    elsif rx_byte = x"45" or rx_byte = x"65" then
                        line_mode <= line_end;
                    elsif is_digit(rx_byte) then
                        line_mode     <= line_box;
                        current_value <= resize(unsigned(rx_byte) - to_unsigned(48, 8), 16);
                        digit_seen    <= '1';
                    elsif not is_separator(rx_byte) then
                        line_mode   <= line_box;
                        parse_error <= '1';
                    end if;

                elsif line_mode = line_box then
                    if is_digit(rx_byte) then
                        digit_value := resize(unsigned(rx_byte) - to_unsigned(48, 8), 16);
                        if current_value > to_unsigned(6553, 16) or
                           (current_value = to_unsigned(6553, 16) and
                            digit_value > to_unsigned(5, 16)) then
                            parse_error <= '1';
                        else
                            new_value := resize((current_value * 10) + digit_value, 16);
                            current_value <= new_value;
                        end if;
                        digit_seen <= '1';

                    elsif is_separator(rx_byte) then
                        if digit_seen = '1' then
                            if field_index < 5 then
                                case field_index is
                                    when 0 => field0 <= current_value;
                                    when 1 => field1 <= current_value;
                                    when 2 => field2 <= current_value;
                                    when 3 => field3 <= current_value;
                                    when others => field4 <= current_value;
                                end case;
                                field_index   <= field_index + 1;
                                current_value <= (others => '0');
                                digit_seen    <= '0';
                            else
                                parse_error <= '1';
                            end if;
                        end if;
                    else
                        parse_error <= '1';
                    end if;

                elsif not is_separator(rx_byte) then
                    parse_error <= '1';
                end if;
            end if;
        end if;
    end process;

    with sw select
        led <= std_logic_vector(display_cell(3 downto 0)) when "00",
               std_logic_vector(display_droplet(3 downto 0)) when "01",
               std_logic_vector(display_cell(3 downto 0) +
                                display_droplet(3 downto 0)) when "10",
               std_logic_vector(display_rejected(3 downto 0)) when others;
end architecture rtl;
