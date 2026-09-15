library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity tb_qnn_cell_radial_guard_filter is
end entity;

architecture sim of tb_qnn_cell_radial_guard_filter is
    constant CLOCK_PERIOD : time := 10 ns;
    signal clk : std_logic := '0';
    signal reset_n : std_logic := '0';
    signal image_tdata : std_logic_vector(7 downto 0) := (others => '0');
    signal image_tvalid : std_logic := '0';
    signal image_tready : std_logic := '1';
    signal input_data : std_logic_vector(7 downto 0) := (others => '0');
    signal input_valid : std_logic := '0';
    signal input_ready : std_logic;
    signal output_data : std_logic_vector(7 downto 0);
    signal output_valid : std_logic;
    signal output_ready : std_logic := '1';
begin
    clk <= not clk after CLOCK_PERIOD / 2;

    dut : entity work.qnn_cell_radial_guard_filter
        port map (
            clk => clk,
            reset_n => reset_n,
            image_tdata => image_tdata,
            image_tvalid => image_tvalid,
            image_tready => image_tready,
            s_axis_tdata => input_data,
            s_axis_tvalid => input_valid,
            s_axis_tready => input_ready,
            m_axis_tdata => output_data,
            m_axis_tvalid => output_valid,
            m_axis_tready => output_ready
        );

    stimulus : process
        procedure send_image_byte(value : natural) is
        begin
            image_tdata <= std_logic_vector(to_unsigned(value, 8));
            image_tvalid <= '1';
            wait until rising_edge(clk);
            image_tvalid <= '0';
        end procedure;

        procedure send_qnn_byte(value : integer) is
        begin
            input_data <= std_logic_vector(to_signed(value, 8));
            input_valid <= '1';
            loop
                wait until rising_edge(clk);
                exit when input_ready = '1';
            end loop;
            input_valid <= '0';
        end procedure;

        procedure expect_output(value : integer) is
        begin
            loop
                wait until rising_edge(clk);
                exit when output_valid = '1';
            end loop;
            assert signed(output_data) = to_signed(value, 8)
                report "Unexpected filtered output byte"
                severity failure;
        end procedure;
    begin
        wait for 5 * CLOCK_PERIOD;
        wait until rising_edge(clk);
        reset_n <= '1';

        -- Fill one 96x96 frame. Pixel (2,2) is the bright center for grid 0.
        for address in 0 to 9215 loop
            if address = 2 * 96 + 2 then
                send_image_byte(200);
            else
                send_image_byte(100);
            end if;
        end loop;

        -- Cell slot 0: confidence code 66 is promoted to 105 by radial support.
        send_qnn_byte(66);
        send_qnn_byte(0);
        send_qnn_byte(0);
        send_qnn_byte(20);
        send_qnn_byte(20);
        expect_output(105);
        expect_output(0);
        expect_output(0);
        expect_output(20);
        expect_output(20);

        -- Cell slot 1 below the low threshold remains unchanged.
        send_qnn_byte(60);
        send_qnn_byte(1);
        send_qnn_byte(2);
        send_qnn_byte(3);
        send_qnn_byte(4);
        expect_output(60);
        expect_output(1);
        expect_output(2);
        expect_output(3);
        expect_output(4);

        -- Droplet slot is never modified by the cell guard.
        send_qnn_byte(66);
        send_qnn_byte(5);
        send_qnn_byte(6);
        send_qnn_byte(7);
        send_qnn_byte(8);
        expect_output(66);
        expect_output(5);
        expect_output(6);
        expect_output(7);
        expect_output(8);

        report "qnn_cell_radial_guard_filter simulation PASS" severity note;
        wait;
    end process;
end architecture sim;
