use strict;
use warnings;

my $w1 = shift; #you want newer word embeddings here XXXL
my $w2 = shift; ## old embeddings to be updated! sl999

open IN1, "< $w1" or die $!;
open IN2, "< $w2" or die $!;

my %word;
my $cc = 0;
while(<IN1>){
	chomp;	
	my @toks = split(/\s+/, $_);
	if($#toks != 300 ){
		print("not possible!" . $#toks);
		exit;
	}
	my $embStr = join(" ", @toks[1 .. $#toks]);
	#print $embStr if $cc == 0;
	if(exists $word{$toks[0]}){
		print("not possible 2");
		exit;
	}
	
	$word{$toks[0]} = $embStr;
	$cc++;
}

my $dd = 0;
while(<IN2>){
	chomp;	
	my @toks = split(/\s+/, $_);
	if($#toks != 300 ){
                print("not possible!" . $#toks);
                exit;
        }	
	my $embStr = join(" ", @toks[1 .. $#toks]);
	if(exists $word{$toks[0]}){
		$dd ++;			
        }else{
		$word{$toks[0]} = $embStr;
	}

}

open OUT, "> paragram_300_sl999_2XXXL.txt" or die $!;

foreach my $name (sort keys %word) {
	print OUT $name." ".$word{$name}."\n";
}

print $cc." output - overlap $dd\n";

close IN1;
close IN2;
close OUT;
