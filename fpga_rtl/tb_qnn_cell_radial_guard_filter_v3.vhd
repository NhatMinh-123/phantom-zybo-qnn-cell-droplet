library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.env.all;

entity tb_qnn_cell_radial_guard_filter_v3 is
end entity;

architecture sim of tb_qnn_cell_radial_guard_filter_v3 is
    signal clk : std_logic := '0';
    signal reset_n : std_logic := '0';
    signal image_data : std_logic_vector(7 downto 0) := (others => '0');
    signal image_valid : std_logic := '0';
    signal input_data : std_logic_vector(7 downto 0) := (others => '0');
    signal input_valid : std_logic := '0';
    signal input_ready : std_logic;
    signal output_data : std_logic_vector(7 downto 0);
    signal output_valid : std_logic;
begin
    clk <= not clk after 5 ns;

    dut : entity work.qnn_cell_radial_guard_filter_v3
        port map (
            clk => clk,
            reset_n => reset_n,
            image_tdata => image_data,
            image_tvalid => image_valid,
            image_tready => '1',
            s_axis_tdata => input_data,
            s_axis_tvalid => input_valid,
            s_axis_tready => input_ready,
            m_axis_tdata => output_data,
            m_axis_tvalid => output_valid,
            m_axis_tready => '1'
        );

    stimulus : process
        procedure send_image(value : natural) is
        begin
            image_data <= std_logic_vector(to_unsigned(value, 8));
            image_valid <= '1';
            wait until rising_edge(clk);
            image_valid <= '0';
        end procedure;
        procedure send_qnn(value : integer) is
        begin
            input_data <= std_logic_vector(to_signed(value, 8));
            input_valid <= '1';
            loop
                wait until rising_edge(clk);
                exit when input_ready = '1';
            end loop;
            input_valid <= '0';
        end procedure;
        procedure expect(value : integer) is
        begin
            loop
                wait until rising_edge(clk);
                exit when output_valid = '1';
            end loop;
            assert signed(output_data) = to_signed(value, 8)
                report "Unexpected guarded byte" severity failure;
        end procedure;
    begin
        wait for 40 ns;
        wait until rising_edge(clk);
        reset_n <= '1';

        -- Present a low-confidence slot before its local image neighborhood is ready.
        for address in 0 to 99 loop
            send_image(100);
        end loop;
        send_qnn(66);
        send_qnn(0);
        send_qnn(0);
        send_qnn(20);
        send_qnn(20);

        -- Grid (0,0) requires pixels only through address 582, not the full frame.
        for address in 100 to 582 loop
            if address = 2 * 96 + 2 then
                send_image(200);
            else
                send_image(100);
            end if;
        end loop;

        expect(105);
        expect(0);
        expect(0);
        expect(20);
        expect(20);

        report "qnn_cell_radial_guard_filter_v3 simulation PASS" severity note;
        stop;
        wait;
    end process;
end architecture sim;
