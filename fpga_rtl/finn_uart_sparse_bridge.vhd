library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity finn_uart_sparse_bridge is
    generic (
        CLKS_PER_BIT         : positive := 9;
        INPUT_BYTES          : positive := 36864;
        OUTPUT_WORDS         : positive := 34560;
        OUTPUT_CHANNELS      : positive := 15;
        GRID_POINTS          : positive := 2304;
        MAX_CANDIDATES       : positive := 256;
        CELL_OBJECT_CODE     : integer := 34;
        DROPLET_OBJECT_CODE  : integer := 43
    );
    port (
        clk     : in  std_logic;
        reset_n : in  std_logic;

        uart_rx_i : in  std_logic;
        uart_tx_o : out std_logic;

        core_reset_n : out std_logic;

        s_axis_tdata  : out std_logic_vector(7 downto 0);
        s_axis_tvalid : out std_logic;
        s_axis_tready : in  std_logic;

        m_axis_tdata  : in  std_logic_vector(7 downto 0);
        m_axis_tvalid : in  std_logic;
        m_axis_tready : out std_logic;

        status_led : out std_logic_vector(3 downto 0)
    );
end entity finn_uart_sparse_bridge;

architecture rtl of finn_uart_sparse_bridge is
    type state_t is (
        sync_c,
        sync_d,
        sync_q,
        sync_one,
        frame_id_low,
        frame_id_high,
        receive_payload,
        checksum_low,
        checksum_high,
        stream_input_prefetch,
        stream_input,
        wait_for_output,
        transmit_header,
        transmit_payload_prefetch,
        transmit_payload_byte,
        transmit_checksum_low,
        transmit_checksum_high
    );
    type tx_phase_t is (tx_launch, tx_wait_busy, tx_wait_done);
    type input_memory_t is array (0 to INPUT_BYTES - 1) of
        std_logic_vector(7 downto 0);
    type candidate_memory_t is array (0 to MAX_CANDIDATES - 1) of
        std_logic_vector(63 downto 0);
    type slot_values_t is array (0 to 4) of std_logic_vector(7 downto 0);

    constant RECORD_BYTES : positive := 8;

    signal state    : state_t := sync_c;
    signal tx_phase : tx_phase_t := tx_launch;

    signal rx_valid : std_logic;
    signal rx_byte  : std_logic_vector(7 downto 0);
    signal tx_start : std_logic := '0';
    signal tx_byte  : std_logic_vector(7 downto 0) := (others => '0');
    signal tx_busy  : std_logic;

    signal input_valid_reg   : std_logic := '0';
    signal input_received    : natural range 0 to INPUT_BYTES := 0;
    signal input_accepted    : natural range 0 to INPUT_BYTES := 0;
    signal input_checksum    : unsigned(15 downto 0) := (others => '0');
    signal expected_checksum : unsigned(15 downto 0) := (others => '0');

    signal input_memory     : input_memory_t;
    signal input_read_addr  : natural range 0 to INPUT_BYTES - 1 := 0;
    signal input_read_data  : std_logic_vector(7 downto 0) := (others => '0');
    signal input_send_index : natural range 0 to INPUT_BYTES - 1 := 0;

    signal candidate_memory    : candidate_memory_t;
    signal candidate_count     : natural range 0 to MAX_CANDIDATES := 0;
    signal candidate_read_addr : natural range 0 to MAX_CANDIDATES - 1 := 0;
    signal candidate_read_data : std_logic_vector(63 downto 0) :=
        (others => '0');
    signal candidate_byte_index : natural range 0 to RECORD_BYTES - 1 := 0;
    signal candidate_overflow   : std_logic := '0';

    signal output_count  : natural range 0 to OUTPUT_WORDS := 0;
    signal channel_index : natural range 0 to OUTPUT_CHANNELS - 1 := 0;
    signal grid_index    : natural range 0 to GRID_POINTS - 1 := 0;
    signal slot_values   : slot_values_t := (others => (others => '0'));

    signal output_checksum : unsigned(15 downto 0) := (others => '0');
    signal frame_id        : std_logic_vector(15 downto 0) := (others => '0');
    signal frame_active    : std_logic := '0';
    signal core_reset_n_reg : std_logic := '0';
    signal response_status  : std_logic_vector(7 downto 0) := (others => '0');
    signal response_has_payload : std_logic := '0';
    signal header_index     : natural range 0 to 14 := 0;
    signal error_latched    : std_logic := '0';
    signal measurement_active : std_logic := '0';
    signal cycle_counter      : unsigned(31 downto 0) := (others => '0');
    signal accelerator_cycles : unsigned(31 downto 0) := (others => '0');

    attribute ram_style : string;
    attribute ram_style of input_memory : signal is "block";
    attribute ram_style of candidate_memory : signal is "block";

    function response_length_byte(
        byte_index  : natural;
        has_payload : std_logic;
        records     : natural
    ) return std_logic_vector is
        variable length_value : natural := 0;
    begin
        if has_payload = '1' then
            length_value := records * RECORD_BYTES;
        end if;
        case byte_index is
            when 0 =>
                return std_logic_vector(to_unsigned(length_value mod 256, 8));
            when 1 =>
                return std_logic_vector(to_unsigned((length_value / 256) mod 256, 8));
            when 2 =>
                return std_logic_vector(to_unsigned((length_value / 65536) mod 256, 8));
            when others =>
                return std_logic_vector(to_unsigned((length_value / 16777216) mod 256, 8));
        end case;
    end function;
begin
    assert OUTPUT_CHANNELS = 15
        report "Sparse bridge expects three five-value detector slots"
        severity failure;
    assert OUTPUT_WORDS = GRID_POINTS * OUTPUT_CHANNELS
        report "OUTPUT_WORDS must equal GRID_POINTS * OUTPUT_CHANNELS"
        severity failure;

    core_reset_n <= core_reset_n_reg and reset_n;
    s_axis_tdata <= input_read_data;
    s_axis_tvalid <= input_valid_reg;
    m_axis_tready <= '1' when frame_active = '1' and
                              output_count < OUTPUT_WORDS else '0';

    status_led(0) <= reset_n;
    status_led(1) <= frame_active;
    status_led(2) <= '1' when state = transmit_header or
                              state = transmit_payload_prefetch or
                              state = transmit_payload_byte or
                              state = transmit_checksum_low or
                              state = transmit_checksum_high else '0';
    status_led(3) <= error_latched;

    rx_inst : entity work.uart_rx
        generic map (CLKS_PER_BIT => CLKS_PER_BIT)
        port map (
            clk       => clk,
            rx_serial => uart_rx_i,
            rx_valid  => rx_valid,
            rx_byte   => rx_byte
        );

    tx_inst : entity work.uart_tx
        generic map (CLKS_PER_BIT => CLKS_PER_BIT)
        port map (
            clk       => clk,
            tx_start  => tx_start,
            tx_byte   => tx_byte,
            tx_serial => uart_tx_o,
            tx_busy   => tx_busy
        );

    process (
        state,
        header_index,
        frame_id,
        response_status,
        response_has_payload,
        candidate_count,
        candidate_read_data,
        candidate_byte_index,
        output_checksum,
        accelerator_cycles
    )
    begin
        tx_byte <= (others => '0');
        case state is
            when transmit_header =>
                case header_index is
                    when 0 => tx_byte <= x"52"; -- R
                    when 1 => tx_byte <= x"44"; -- D
                    when 2 => tx_byte <= x"53"; -- S
                    when 3 => tx_byte <= x"31"; -- 1
                    when 4 => tx_byte <= frame_id(7 downto 0);
                    when 5 => tx_byte <= frame_id(15 downto 8);
                    when 6 => tx_byte <= response_status;
                    when 7 =>
                        tx_byte <= response_length_byte(
                            0, response_has_payload, candidate_count
                        );
                    when 8 =>
                        tx_byte <= response_length_byte(
                            1, response_has_payload, candidate_count
                        );
                    when 9 =>
                        tx_byte <= response_length_byte(
                            2, response_has_payload, candidate_count
                        );
                    when 10 =>
                        tx_byte <= response_length_byte(
                            3, response_has_payload, candidate_count
                        );
                    when 11 =>
                        tx_byte <= std_logic_vector(accelerator_cycles(7 downto 0));
                    when 12 =>
                        tx_byte <= std_logic_vector(accelerator_cycles(15 downto 8));
                    when 13 =>
                        tx_byte <= std_logic_vector(accelerator_cycles(23 downto 16));
                    when others =>
                        tx_byte <= std_logic_vector(accelerator_cycles(31 downto 24));
                end case;
            when transmit_payload_byte =>
                tx_byte <= candidate_read_data(
                    8 * candidate_byte_index + 7 downto
                    8 * candidate_byte_index
                );
            when transmit_checksum_low =>
                tx_byte <= std_logic_vector(output_checksum(7 downto 0));
            when transmit_checksum_high =>
                tx_byte <= std_logic_vector(output_checksum(15 downto 8));
            when others =>
                null;
        end case;
    end process;

    process (clk)
    begin
        if rising_edge(clk) then
            input_read_data <= input_memory(input_read_addr);
            if state = receive_payload and rx_valid = '1' and
               input_received < INPUT_BYTES then
                input_memory(input_received) <= rx_byte;
            end if;
            candidate_read_data <= candidate_memory(candidate_read_addr);
        end if;
    end process;

    process (clk)
        variable accepted_input  : boolean;
        variable accepted_output : boolean;
        variable checksum_candidate : unsigned(15 downto 0);
        variable byte_sum : unsigned(16 downto 0);
        variable field_index : natural range 0 to 4;
        variable slot_index  : natural range 0 to 2;
        variable object_code : signed(7 downto 0);
        variable threshold_code : signed(7 downto 0);
        variable grid_value : unsigned(15 downto 0);
        variable record_value : std_logic_vector(63 downto 0);
    begin
        if rising_edge(clk) then
            tx_start <= '0';

            if reset_n = '0' then
                state                 <= sync_c;
                tx_phase              <= tx_launch;
                input_valid_reg       <= '0';
                input_received        <= 0;
                input_accepted        <= 0;
                input_checksum        <= (others => '0');
                expected_checksum     <= (others => '0');
                input_read_addr       <= 0;
                input_send_index      <= 0;
                candidate_count       <= 0;
                candidate_read_addr   <= 0;
                candidate_byte_index  <= 0;
                candidate_overflow    <= '0';
                output_count          <= 0;
                channel_index         <= 0;
                grid_index            <= 0;
                slot_values           <= (others => (others => '0'));
                output_checksum       <= (others => '0');
                frame_id              <= (others => '0');
                frame_active          <= '0';
                core_reset_n_reg      <= '0';
                response_status       <= (others => '0');
                response_has_payload  <= '0';
                header_index          <= 0;
                error_latched         <= '0';
                measurement_active    <= '0';
                cycle_counter         <= (others => '0');
                accelerator_cycles    <= (others => '0');
            else
                accepted_input := input_valid_reg = '1' and s_axis_tready = '1';
                accepted_output := frame_active = '1' and
                    output_count < OUTPUT_WORDS and m_axis_tvalid = '1';

                if accepted_input then
                    if input_accepted < INPUT_BYTES then
                        input_accepted <= input_accepted + 1;
                    end if;
                    if input_accepted = 0 then
                        measurement_active <= '1';
                        cycle_counter <= (others => '0');
                    end if;
                end if;
                if measurement_active = '1' then
                    cycle_counter <= cycle_counter + 1;
                end if;

                if accepted_output then
                    output_count <= output_count + 1;
                    field_index := channel_index mod 5;
                    slot_index := channel_index / 5;
                    slot_values(field_index) <= m_axis_tdata;

                    if field_index = 4 then
                        object_code := signed(slot_values(0));
                        if slot_index < 2 then
                            threshold_code := to_signed(CELL_OBJECT_CODE, 8);
                        else
                            threshold_code := to_signed(DROPLET_OBJECT_CODE, 8);
                        end if;
                        if object_code >= threshold_code then
                            if candidate_count < MAX_CANDIDATES then
                                grid_value := to_unsigned(grid_index, 16);
                                record_value := (others => '0');
                                record_value(7 downto 0) :=
                                    std_logic_vector(grid_value(7 downto 0));
                                record_value(15 downto 8) :=
                                    std_logic_vector(grid_value(15 downto 8));
                                record_value(23 downto 16) :=
                                    std_logic_vector(to_unsigned(slot_index, 8));
                                record_value(31 downto 24) := slot_values(0);
                                record_value(39 downto 32) := slot_values(1);
                                record_value(47 downto 40) := slot_values(2);
                                record_value(55 downto 48) := slot_values(3);
                                record_value(63 downto 56) := m_axis_tdata;
                                candidate_memory(candidate_count) <= record_value;
                                candidate_count <= candidate_count + 1;
                            else
                                candidate_overflow <= '1';
                            end if;
                        end if;
                    end if;

                    if channel_index = OUTPUT_CHANNELS - 1 then
                        channel_index <= 0;
                        if grid_index < GRID_POINTS - 1 then
                            grid_index <= grid_index + 1;
                        end if;
                    else
                        channel_index <= channel_index + 1;
                    end if;

                    if output_count = OUTPUT_WORDS - 1 then
                        if measurement_active = '1' then
                            accelerator_cycles <= cycle_counter + 1;
                        else
                            accelerator_cycles <= (others => '0');
                        end if;
                        measurement_active <= '0';
                    end if;
                end if;

                case state is
                    when sync_c =>
                        core_reset_n_reg <= '0';
                        frame_active <= '0';
                        if rx_valid = '1' and rx_byte = x"43" then
                            state <= sync_d;
                        end if;

                    when sync_d =>
                        if rx_valid = '1' then
                            if rx_byte = x"44" then
                                state <= sync_q;
                            elsif rx_byte /= x"43" then
                                state <= sync_c;
                            end if;
                        end if;

                    when sync_q =>
                        if rx_valid = '1' then
                            if rx_byte = x"51" then
                                state <= sync_one;
                            elsif rx_byte = x"43" then
                                state <= sync_d;
                            else
                                state <= sync_c;
                            end if;
                        end if;

                    when sync_one =>
                        if rx_valid = '1' then
                            if rx_byte = x"31" then
                                input_received       <= 0;
                                input_accepted       <= 0;
                                input_valid_reg      <= '0';
                                input_checksum       <= (others => '0');
                                expected_checksum    <= (others => '0');
                                input_read_addr      <= 0;
                                input_send_index     <= 0;
                                candidate_count      <= 0;
                                candidate_read_addr  <= 0;
                                candidate_byte_index <= 0;
                                candidate_overflow   <= '0';
                                output_count         <= 0;
                                channel_index        <= 0;
                                grid_index           <= 0;
                                slot_values          <= (others => (others => '0'));
                                output_checksum      <= (others => '0');
                                response_status      <= (others => '0');
                                response_has_payload <= '0';
                                header_index         <= 0;
                                measurement_active   <= '0';
                                cycle_counter        <= (others => '0');
                                accelerator_cycles   <= (others => '0');
                                state                <= frame_id_low;
                            elsif rx_byte = x"43" then
                                state <= sync_d;
                            else
                                state <= sync_c;
                            end if;
                        end if;

                    when frame_id_low =>
                        if rx_valid = '1' then
                            frame_id(7 downto 0) <= rx_byte;
                            state <= frame_id_high;
                        end if;

                    when frame_id_high =>
                        if rx_valid = '1' then
                            frame_id(15 downto 8) <= rx_byte;
                            state <= receive_payload;
                        end if;

                    when receive_payload =>
                        if rx_valid = '1' then
                            input_checksum <= input_checksum +
                                resize(unsigned(rx_byte), 16);
                            if input_received = INPUT_BYTES - 1 then
                                input_received <= INPUT_BYTES;
                                state <= checksum_low;
                            else
                                input_received <= input_received + 1;
                            end if;
                        end if;

                    when checksum_low =>
                        if rx_valid = '1' then
                            expected_checksum(7 downto 0) <= unsigned(rx_byte);
                            state <= checksum_high;
                        end if;

                    when checksum_high =>
                        if rx_valid = '1' then
                            checksum_candidate :=
                                unsigned(rx_byte) & expected_checksum(7 downto 0);
                            expected_checksum(15 downto 8) <= unsigned(rx_byte);
                            if checksum_candidate /= input_checksum then
                                response_status <= x"01";
                                response_has_payload <= '0';
                                error_latched <= '1';
                                frame_active <= '0';
                                core_reset_n_reg <= '0';
                                header_index <= 0;
                                tx_phase <= tx_launch;
                                state <= transmit_header;
                            else
                                frame_active <= '1';
                                core_reset_n_reg <= '1';
                                input_read_addr <= 0;
                                state <= stream_input_prefetch;
                            end if;
                        end if;

                    when stream_input_prefetch =>
                        input_valid_reg <= '1';
                        state <= stream_input;

                    when stream_input =>
                        if accepted_input then
                            if input_send_index = INPUT_BYTES - 1 then
                                input_valid_reg <= '0';
                                state <= wait_for_output;
                            else
                                input_valid_reg <= '0';
                                input_send_index <= input_send_index + 1;
                                input_read_addr <= input_send_index + 1;
                                state <= stream_input_prefetch;
                            end if;
                        end if;

                    when wait_for_output =>
                        if input_valid_reg = '0' and
                           input_accepted = INPUT_BYTES and
                           output_count = OUTPUT_WORDS then
                            frame_active <= '0';
                            header_index <= 0;
                            tx_phase <= tx_launch;
                            if candidate_overflow = '1' then
                                response_status <= x"04";
                                response_has_payload <= '0';
                                error_latched <= '1';
                            else
                                response_status <= x"00";
                                response_has_payload <= '1';
                            end if;
                            state <= transmit_header;
                        elsif input_valid_reg = '0' and
                              input_received = INPUT_BYTES and
                              input_accepted < INPUT_BYTES then
                            response_status <= x"03";
                            response_has_payload <= '0';
                            error_latched <= '1';
                            frame_active <= '0';
                            core_reset_n_reg <= '0';
                            header_index <= 0;
                            tx_phase <= tx_launch;
                            state <= transmit_header;
                        end if;

                    when transmit_payload_prefetch =>
                        candidate_byte_index <= 0;
                        state <= transmit_payload_byte;
                        tx_phase <= tx_launch;

                    when transmit_header | transmit_payload_byte |
                         transmit_checksum_low | transmit_checksum_high =>
                        case tx_phase is
                            when tx_launch =>
                                tx_start <= '1';
                                tx_phase <= tx_wait_busy;
                            when tx_wait_busy =>
                                if tx_busy = '1' then
                                    tx_phase <= tx_wait_done;
                                end if;
                            when tx_wait_done =>
                                if tx_busy = '0' then
                                    tx_phase <= tx_launch;
                                    case state is
                                        when transmit_header =>
                                            if header_index = 14 then
                                                if response_has_payload = '1' and
                                                   candidate_count > 0 then
                                                    candidate_read_addr <= 0;
                                                    state <= transmit_payload_prefetch;
                                                else
                                                    output_checksum <= (others => '0');
                                                    state <= transmit_checksum_low;
                                                end if;
                                            else
                                                header_index <= header_index + 1;
                                            end if;
                                        when transmit_payload_byte =>
                                            byte_sum := resize(output_checksum, 17) +
                                                resize(unsigned(candidate_read_data(
                                                    8 * candidate_byte_index + 7 downto
                                                    8 * candidate_byte_index
                                                )), 17);
                                            output_checksum <= byte_sum(15 downto 0);
                                            if candidate_byte_index = RECORD_BYTES - 1 then
                                                if candidate_read_addr =
                                                   candidate_count - 1 then
                                                    state <= transmit_checksum_low;
                                                else
                                                    candidate_read_addr <=
                                                        candidate_read_addr + 1;
                                                    state <= transmit_payload_prefetch;
                                                end if;
                                            else
                                                candidate_byte_index <=
                                                    candidate_byte_index + 1;
                                            end if;
                                        when transmit_checksum_low =>
                                            state <= transmit_checksum_high;
                                        when transmit_checksum_high =>
                                            core_reset_n_reg <= '0';
                                            response_has_payload <= '0';
                                            state <= sync_c;
                                        when others =>
                                            null;
                                    end case;
                                end if;
                        end case;
                end case;
            end if;
        end if;
    end process;
end architecture rtl;
