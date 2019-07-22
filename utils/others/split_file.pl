#!/usr/bin/env perl
# The code is writing according to kaldi/egs/wsj/s5/utils/split_scp.pl
# author : chenhangting
use warnings; #sed replacement for -w perl parameter

if (@ARGV < 4){
    die "Usage: split_file.pl line_num_block in.scp out1.scp out2.scp ... \n";
}

$block_size = shift @ARGV;
$inscp = shift @ARGV;
@OUTPUTS = @ARGV;

$numscps = @OUTPUTS;  # size of array.
@F = ();

open(I, "<$inscp") || die "Opening input scp file $inscp";
while(<I>) {
    push @F, $_;
}
$numlines = @F;
$numlines = int($numlines / $block_size);

if($numlines == 0){
    die "split_scp.pl: error: empty input scp file";
}

$linesperscp = int( $numlines / $numscps ); # the "whole part"..
$remainder = $numlines - ($linesperscp * $numscps);
($remainder >= 0 && $remainder < $numlines) || die "bad remainder $remainder";
$n=0;
for($scpidx = 0; $scpidx < @OUTPUTS; $scpidx++) {
    $scpfile = $OUTPUTS[$scpidx];
    open(O, ">$scpfile") || die "Opening output scp file $scpfile";
    for($k = 0; $k < $linesperscp + ($scpidx < $remainder ? 1 : 0); $k++) {
        for($m = 0; $m < $block_size; $m++){
            print O $F[$n++];
        }
    }
    close(O) || die "Closing scp file $scpfile";
}
$n == $numlines * $block_size || die "split_scp.pl: code error., $n != $numlines";
