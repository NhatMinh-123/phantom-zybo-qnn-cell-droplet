library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity qnn_cell_radial_guard_filter is
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
end entity qnn_cell_radial_guard_filter;

architecture rtl of qnn_cell_radial_guard_filter is
    constant IMAGE_PIXELS  : positive := IMAGE_WIDTH * IMAGE_HEIGHT;
    constant GRID_POINTS   : positive := GRID_WIDTH * GRID_HEIGHT;
    constant SUPPORT_SIDE  : positive := 2 * SUPPORT_RADIUS + 1;
    constant SUPPORT_POINTS : positive := SUPPORT_SIDE * SUPPORT_SIDE;

    type image_memory_t is array (0 to IMAGE_PIXELS - 1) of
        std_logic_vector(7 downto 0);
    type slot_buffer_t is array (0 to 4) of std_logic_vector(7 downto 0);
    type state_t is (collect_slot, scan_issue, scan_wait, scan_consume, emit_slot);

    signal image_memory : image_memory_t;
    signal image_write_address : natural range 0 to IMAGE_PIXELS - 1 := 0;
    signal image_count : natural range 0 to IMAGE_PIXELS := 0;
    signal image_read_address : natural range 0 to IMAGE_PIXELS - 1 := 0;
    signal image_read_data : std_logic_vector(7 downto 0) := (others => '0');

    signal state : state_t := collect_slot;
    signal slot_buffer : slot_buffer_t := (others => (others => '0'));
    signal collect_field : natural range 0 to 4 := 0;
    signal emit_field : natural range 0 to 4 := 0;
    signal slot_index : natural range 0 to 2 := 0;
    signal grid_index : natural range 0 to GRID_POINTS - 1 := 0;
    signal support_index : natural range 0 to SUPPORT_POINTS - 1 := 0;
    signal sample_index : natural range 0 to 8 := 0;
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
    assert GRID_WIDTH * GRID_STRIDE = IMAGE_WIDTH
        report "Grid width and stride must span the input image"
        severity failure;
    assert GRID_HEIGHT * GRID_STRIDE = IMAGE_HEIGHT
        report "Grid height and stride must span the input image"
        severity failure;
    assert CELL_LOW_OBJECT_CODE < CELL_HIGH_OBJECT_CODE
        report "Low cell threshold must be below the high threshold"
        severity failure;

    s_axis_tready <= '1' when state = collect_slot else '0';
    m_axis_tvalid <= '1' when state = emit_slot else '0';
    m_axis_tdata <= std_logic_vector(to_signed(CELL_HIGH_OBJECT_CODE, 8))
        when state = emit_slot and emit_field = 0 and promote_cell = '1'
        else slot_buffer(emit_field);

    -- Port A writes the exact grayscale bytes accepted by the QNN.
    process (clk)
    begin
        if rising_edge(clk) then
            if reset_n = '0' then
                image_write_address <= 0;
                image_count <= 0;
            elsif image_tvalid = '1' and image_tready = '1' then
                image_memory(image_write_address) <= image_tdata;
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

    -- Port B is a synchronous read port used only for guarded low-confidence cells.
    process (clk)
    begin
        if rising_edge(clk) then
            image_read_data <= image_memory(image_read_address);
        end if;
    end process;

    process (clk)
        variable object_code : signed(7 downto 0);
        variable base_x : integer;
        variable base_y : integer;
        variable point_x : integer;
        variable point_y : integer;
        variable sample_x : integer;
        variable sample_y : integer;
        variable required_x : integer;
        variable required_y : integer;
        variable required_address : integer;
        variable sample_value : integer;
        variable final_response : integer;
    begin
        if rising_edge(clk) then
            if reset_n = '0' then
                state <= collect_slot;
                slot_buffer <= (others => (others => '0'));
                collect_field <= 0;
                emit_field <= 0;
                slot_index <= 0;
                grid_index <= 0;
                support_index <= 0;
                sample_index <= 0;
                radial_accumulator <= 0;
                promote_cell <= '0';
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
                                    base_x := (grid_index mod GRID_WIDTH) * GRID_STRIDE +
                                        GRID_CENTER_OFFSET;
                                    base_y := (grid_index / GRID_WIDTH) * GRID_STRIDE +
                                        GRID_CENTER_OFFSET;
                                    required_x := clamp(
                                        base_x + integer(SUPPORT_RADIUS) + 3,
                                        0, IMAGE_WIDTH - 1
                                    );
                                    required_y := clamp(
                                        base_y + integer(SUPPORT_RADIUS) + 3,
                                        0, IMAGE_HEIGHT - 1
                                    );
                                    required_address := required_y * IMAGE_WIDTH + required_x;
                                    -- Never stall the QNN waiting for future image bytes.
                                    -- If a pathological pipeline emits too early, reject only
                                    -- that low-confidence proposal and preserve high-confidence data.
                                    if required_address < image_count then
                                        support_index <= 0;
                                        sample_index <= 0;
                                        radial_accumulator <= 0;
                                        state <= scan_issue;
                                    else
                                        state <= emit_slot;
                                    end if;
                                else
                                    state <= emit_slot;
                                end if;
                            else
                                collect_field <= collect_field + 1;
                            end if;
                        end if;

                    when scan_issue =>
                        base_x := (grid_index mod GRID_WIDTH) * GRID_STRIDE +
                            GRID_CENTER_OFFSET;
                        base_y := (grid_index / GRID_WIDTH) * GRID_STRIDE +
                            GRID_CENTER_OFFSET;
                        point_x := base_x + integer(support_index mod SUPPORT_SIDE) -
                            integer(SUPPORT_RADIUS);
                        point_y := base_y + integer(support_index / SUPPORT_SIDE) -
                            integer(SUPPORT_RADIUS);
                        sample_x := clamp(
                            point_x + sample_offset_x(sample_index), 0, IMAGE_WIDTH - 1
                        );
                        sample_y := clamp(
                            point_y + sample_offset_y(sample_index), 0, IMAGE_HEIGHT - 1
                        );
                        image_read_address <= sample_y * IMAGE_WIDTH + sample_x;
                        state <= scan_wait;

                    when scan_wait =>
                        state <= scan_consume;

                    when scan_consume =>
                        sample_value := to_integer(unsigned(image_read_data));
                        if sample_index = 0 then
                            radial_accumulator <= 8 * sample_value;
                            sample_index <= 1;
                            state <= scan_issue;
                        elsif sample_index < 8 then
                            radial_accumulator <= radial_accumulator - sample_value;
                            sample_index <= sample_index + 1;
                            state <= scan_issue;
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
                                state <= scan_issue;
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
                                    if grid_index = GRID_POINTS - 1 then
                                        grid_index <= 0;
                                    else
                                        grid_index <= grid_index + 1;
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
