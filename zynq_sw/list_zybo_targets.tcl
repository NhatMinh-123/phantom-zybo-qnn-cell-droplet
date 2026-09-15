connect
puts "TARGET_LIST=[targets]"
foreach target [targets] {
    targets -set $target
    puts "TARGET=$target NAME=[targets -get-name]"
}
