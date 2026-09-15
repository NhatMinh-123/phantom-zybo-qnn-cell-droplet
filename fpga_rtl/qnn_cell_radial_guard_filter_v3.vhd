library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity qnn_cell_radial_guard_filter_v3 is
    generic (
        IMAGE_WIDTH             : positive := 96;
        IMAGE_HEIGHT            : positive := 96;
        GRID_WIDTH              : positive := 24;
        GRID_HEIGHT             : positive := 24;
        GRID_STRIDE             : positive := 4;
        GRID_CENTER_OFFSET      : natural  := 2;
        CELL_LOW_OBJECT_CODE    : integer  := 66;
        CELL_HIGH_OBJECT_CODE   : integer  := 105;
        RADIAL_THRESHOLD_X8     : integer  := 224;
        SUPPORT_RADIUS          : natural  := 1
    );
    port (
        clk     : in std_logic;
        reset_n : in std_logic;

        image_tdata  : in std_logic_vector(7 downto 0);
        image_tvalid : in std_logic;
        image_tready : in std_logic;

        s_axis_tdata  : in  std_logic_vector(7 downto 0);
        s_axis_tvalid : in  std_logic;
        s_axis_tready : out std_logic;

        m_axis_tdata  : out std_logic_vector(7 downto 0);
        m_axis_tvalid : out std_logic;
        m_axis_tready : in  std_logic
    );
end entity qnn_cell_radial_guard_filter_v3;

architecture rtl of qnn_cell_radial_guard_filter_v3 is
    constant IMAGE_PIXELS   : positive := IMAGE_WIDTH * IMAGE_HEIGHT;
    constant SUPPORT_SIDE   : positive := 2 * SUPPORT_RADIUS + 1;
    constant SUPPORT_POINTS : positive := SUPPORT_SIDE * SUPPORT_SIDE;
    constant RADIAL_RADIUS  : positive := 3;

    type image_memory_t is array (0 to IMAGE_PIXELS - 1) of
        std_logic_vector(7 downto 0);
    type slot_buffer_t is array (0 to 4) of std_logic_vector(7 downto 0);
    type state_t is (
        collect_slot,
        wait_image_ready,
        scan_point_prepare,
        scan_sample_prepare,
        scan_issue,
        scan_wait,
        scan_consume,
        emit_slot
    );

    signal image_memory : image_memory_t;
    signal image_write_address : natural range 0 to IMAGE_PIXELS - 1 := 0;
    signal image_count : natural range 0 to IMAGE_PIXELS := 0;
    signal required_image_count : natural range 0 to IMAGE_PIXELS := 0;
    signal image_read_address : natural range 0 to IMAGE_PIXELS - 1 := 0;
    signal image_read_data : std_logic_vector(7 downto 0) := (others => '0');

    signal state : state_t := collect_slot;
    signal slot_buffer : slot_buffer_t := (others => (others => '0'));
    signal collect_field : natural range 0 to 4 := 0;
    signal emit_field : natural range 0 to 4 := 0;
    signal slot_index : natural range 0 to 2 := 0;
    signal grid_x : natural range 0 to GRID_WIDTH - 1 := 0;
    signal grid_y : natural range 0 to GRID_HEIGHT - 1 := 0;
    signal support_index : natural range 0 to SUPPORT_POINTS - 1 := 0;
    signal sample_index : natural range 0 to 8 := 0;
    signal scan_base_x : natural range 0 to IMAGE_WIDTH - 1 := 0;
    signal scan_base_y : natural range 0 to IMAGE_HEIGHT - 1 := 0;
    signal point_x_reg : natural range 0 to IMAGE_WIDTH - 1 := 0;
    signal point_y_reg : natural range 0 to IMAGE_HEIGHT - 1 := 0;
    signal sample_x_reg : natural range 0 to IMAGE_WIDTH - 1 := 0;
    signal sample_y_reg : natural range 0 to IMAGE_HEIGHT - 1 := 0;
    signal radial_accumulator : integer range -4096 to 4095 := 0;
    signal promote_cell : std_logic := '0';

    attribute ram_style : string;
    attribute ram_style of image_memory : signal is "block";

    function clamp(value : integer; low_value : integer; high_value : integer)
        return integer is
    begin
        if value < low_value then
            return low_value;
        elsif value > high_value then
            return high_value;
        end if;
        return value;
    end function;

    function sample_offset_x(index : natural) return integer is
    begin
        case index is
            when 0 | 1 | 2 => return 0;
            when 3         => return -3;
            when 4         => return 3;
            when 5 | 7     => return -2;
            when others    => return 2;
        end case;
    end function;

    function sample_offset_y(index : natural) return integer is
    begin
        case index is
            when 0 | 3 | 4 => return 0;
            when 1         => return -3;
            when 2         => return 3;
            when 5 | 6     => return -2;
            when others    => return 2;
        end case;
    end function;
begin
    assert IMAGE_WIDTH = 96
        report "Version 3 address pipeline is specialized for 96-pixel rows"
        severity failure;
    assert GRID_WIDTH * GRID_STRIDE = IMAGE_WIDTH
        report "Grid width and stride must span the image"
        severity failure;
    assert GRID_HEIGHT * GRID_STRIDE = IMAGE_HEIGHT
        report "Grid height and stride must span the image"
        severity failure;

    s_axis_tready <= '1' when state = collect_slot else '0';
    m_axis_tvalid <= '1' when state = emit_slot else '0';
    m_axis_tdata <= std_logic_vector(to_signed(CELL_HIGH_OBJECT_CODE, 8))
        when state = emit_slot and emit_field = 0 and promote_cell = '1'
        else slot_buffer(emit_field);

    process (clk)
    begin
        if rising_edge(clk) then
            if image_tvalid = '1' and image_tready = '1' then
                image_memory(image_write_address) <= image_tdata;
            end if;
        end if;
    end process;

    process (clk)
    begin
        if rising_edge(clk) then
            if reset_n = '0' then
                image_write_address <= 0;
                image_count <= 0;
            elsif image_tvalid = '1' and image_tready = '1' then
                if image_count < IMAGE_PIXELS then
                    image_count <= image_count + 1;
                end if;
                if image_write_address = IMAGE_PIXELS - 1 then
                    image_write_address <= 0;
                else
                    image_write_address <= image_write_address + 1;
                end if;
            end if;
        end if;
    end process;

    process (clk)
    begin
        if rising_edge(clk) then
            image_read_data <= image_memory(image_read_address);
        end if;
    end process;

    process (clk)
        variable object_code : signed(7 downto 0);
        variable point_x_value : integer;
        variable point_y_value : integer;
        variable sample_x_value : integer;
        variable sample_y_value : integer;
        variable sample_value : integer;
        variable final_response : integer;
        variable required_x_value : integer;
        variable required_y_value : integer;
        variable required_count_value : integer;
        variable row_times_64 : unsigned(13 downto 0);
        variable row_times_32 : unsigned(13 downto 0);
        variable address_value : unsigned(13 downto 0);
    begin
        if rising_edge(clk) then
            if reset_n = '0' then
                state <= collect_slot;
                slot_buffer <= (others => (others => '0'));
                collect_field <= 0;
                emit_field <= 0;
                slot_index <= 0;
                grid_x <= 0;
                grid_y <= 0;
                support_index <= 0;
                sample_index <= 0;
                scan_base_x <= 0;
                scan_base_y <= 0;
                point_x_reg <= 0;
                point_y_reg <= 0;
                sample_x_reg <= 0;
                sample_y_reg <= 0;
                radial_accumulator <= 0;
                promote_cell <= '0';
                required_image_count <= 0;
                image_read_address <= 0;
            else
                case state is
                    when collect_slot =>
                        if s_axis_tvalid = '1' then
                            slot_buffer(collect_field) <= s_axis_tdata;
                            if collect_field = 4 then
                                collect_field <= 0;
                                emit_field <= 0;
                                promote_cell <= '0';
                                object_code := signed(slot_buffer(0));
                                if slot_index < 2 and
                                   object_code >= to_signed(CELL_LOW_OBJECT_CODE, 8) and
                                   object_code < to_signed(CELL_HIGH_OBJECT_CODE, 8) then
                                    scan_base_x <= grid_x * GRID_STRIDE + GRID_CENTER_OFFSET;
                                    scan_base_y <= grid_y * GRID_STRIDE + GRID_CENTER_OFFSET;
                                    support_index <= 0;
                                    sample_index <= 0;
                                    radial_accumulator <= 0;

                                    required_x_value := clamp(
                                        grid_x * GRID_STRIDE + GRID_CENTER_OFFSET +
                                        integer(SUPPORT_RADIUS) + RADIAL_RADIUS,
                                        0,
                                        IMAGE_WIDTH - 1
                                    );
                                    required_y_value := clamp(
                                        grid_y * GRID_STRIDE + GRID_CENTER_OFFSET +
                                        integer(SUPPORT_RADIUS) + RADIAL_RADIUS,
                                        0,
                                        IMAGE_HEIGHT - 1
                                    );
                                    required_count_value :=
                                        required_y_value * IMAGE_WIDTH + required_x_value + 1;
                                    required_image_count <= required_count_value;
                                    if image_count >= required_count_value then
                                        state <= scan_point_prepare;
                                    else
                                        state <= wait_image_ready;
                                    end if;
                                else
                                    state <= emit_slot;
                                end if;
                            else
                                collect_field <= collect_field + 1;
                            end if;
                        end if;

                    when wait_image_ready =>
                        if image_count >= required_image_count then
                            state <= scan_point_prepare;
                        end if;

                    when scan_point_prepare =>
                        point_x_value := scan_base_x +
                            integer(support_index mod SUPPORT_SIDE) -
                            integer(SUPPORT_RADIUS);
                        point_y_value := scan_base_y +
                            integer(support_index / SUPPORT_SIDE) -
                            integer(SUPPORT_RADIUS);
                        point_x_reg <= clamp(point_x_value, 0, IMAGE_WIDTH - 1);
                        point_y_reg <= clamp(point_y_value, 0, IMAGE_HEIGHT - 1);
                        state <= scan_sample_prepare;

                    when scan_sample_prepare =>
                        sample_x_value := point_x_reg + sample_offset_x(sample_index);
                        sample_y_value := point_y_reg + sample_offset_y(sample_index);
                        sample_x_reg <= clamp(sample_x_value, 0, IMAGE_WIDTH - 1);
                        sample_y_reg <= clamp(sample_y_value, 0, IMAGE_HEIGHT - 1);
                        state <= scan_issue;

                    when scan_issue =>
                        row_times_64 := shift_left(to_unsigned(sample_y_reg, 14), 6);
                        row_times_32 := shift_left(to_unsigned(sample_y_reg, 14), 5);
                        address_value := row_times_64 + row_times_32 +
                            to_unsigned(sample_x_reg, 14);
                        image_read_address <= to_integer(address_value);
                        state <= scan_wait;

                    when scan_wait =>
                        state <= scan_consume;

                    when scan_consume =>
                        sample_value := to_integer(unsigned(image_read_data));
                        if sample_index = 0 then
                            radial_accumulator <= 8 * sample_value;
                            sample_index <= 1;
                            state <= scan_sample_prepare;
                        elsif sample_index < 8 then
                            radial_accumulator <= radial_accumulator - sample_value;
                            sample_index <= sample_index + 1;
                            state <= scan_sample_prepare;
                        else
                            final_response := radial_accumulator - sample_value;
                            if final_response >= RADIAL_THRESHOLD_X8 then
                                promote_cell <= '1';
                                state <= emit_slot;
                            elsif support_index = SUPPORT_POINTS - 1 then
                                state <= emit_slot;
                            else
                                support_index <= support_index + 1;
                                sample_index <= 0;
                                radial_accumulator <= 0;
                                state <= scan_point_prepare;
                            end if;
                        end if;

                    when emit_slot =>
                        if m_axis_tready = '1' then
                            if emit_field = 4 then
                                emit_field <= 0;
                                promote_cell <= '0';
                                state <= collect_slot;
                                if slot_index = 2 then
                                    slot_index <= 0;
                                    if grid_x = GRID_WIDTH - 1 then
                                        grid_x <= 0;
                                        if grid_y = GRID_HEIGHT - 1 then
                                            grid_y <= 0;
                                        else
                                            grid_y <= grid_y + 1;
                                        end if;
                                    else
                                        grid_x <= grid_x + 1;
                                    end if;
                                else
                                    slot_index <= slot_index + 1;
                                end if;
                            else
                                emit_field <= emit_field + 1;
                            end if;
                        end if;
                end case;
            end if;
        end if;
    end process;
end architecture rtl;
