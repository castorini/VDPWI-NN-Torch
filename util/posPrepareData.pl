use strict;
use warnings;

my $posFile = shift;
my $toksFile = shift;

open IN1, "< $posFile"  or die $!;
open IN2, "< $toksFile"  or die $!;

my @tokfiles = <IN2>;
chomp(@tokfiles);

my @posfiles = <IN1>;
chomp(@posfiles);

if(@posfiles != @tokfiles){
	print("How can this be possible?");
	exit;	
}

close IN1;
close IN2;

for(my $i = 0; $i < @posfiles; $i++){
	my $str = "";
	my @poses = split(/\s+/, $posfiles[$i]);
	my @tokses = split(/\s+/, $tokfiles[$i]);
	if(@poses != @tokses){
		print($posfiles[$i]."\n");
		print($tokfiles[$i]."\n");
		foreach(@poses){
			print $_."\n";
		}
		foreach(@tokses){
			print $_."\n";
		}
		print("not possible size not match?!".@poses." ".@tokses);
		exit;
	}
	
	foreach my $poo (@poses){
		my @parts = split(/\_/, $poo);
		#if(@parts != 2){
			#print($posfiles[$i]."\n");
			#print("really: ");
			#print(join(" ",@parts));
			#sleep(5);			
			##exit;
		#}
		$str = $str.$parts[-1]." ";			
	}
	$str =~ s/^\s+|\s+$//g;
	print $str."\n";
}
