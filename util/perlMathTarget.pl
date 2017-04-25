use strict;
use warnings;

use Math::Mathematica;
my $math = Math::Mathematica->new;
#my $result = $math->evaluate('Integrate[Sin[x],{x,0,Pi}]'); # 2

my $task = shift;
my $set = shift;

open IN, "< /cliphomes/huah/deepQuery/torch-ntm/data/".$task."/$set/sim.txt" or die $!;

open OUT, "> /cliphomes/huah/deepQuery/torch-ntm/data/".$task."/$set/sim.sparse.txt" or die $!;


my $labelNo = 6;
my $starter = 1;
if($task eq 'sick'){
    $labelNo = 5;
	$starter = 1;
}
elsif ($task eq 'msrvid' or $task eq 'seme' or $task eq  'smteur' or $task eq 'msrpar'
          or $task eq 'STS2014' or $task eq 'STS2013' or $task eq 'STS2012'){
    $labelNo = 5;
	$starter = 0;
}
elsif($task eq 'twitter' or $task eq 'mspr'){
    $labelNo = 1;
	$starter = 0;
} 
else{
	print("not task");
	exit;
}
  
my @yoshi = <IN>;
chomp(@yoshi);
my $lambda = 0.03;

my $cc = 0;
foreach my $sim (@yoshi){
	my $ceil = int($sim + 0.99);
	my $floor = int($sim);
	
	my $command1 = "0"; #==1
	my $command2 = "0"; #==3.5
	my $command3 = "{0"; #{ITEMS}
	my $i =0;	
	my $pos = 1;
    if($ceil == $floor){
		for($i =$starter; $i <= $labelNo; $i++){
			if ($i == $ceil){
				$command3 = $command3." ,a ";
				$command1 = $command1." + a";
				$command2 = $command2." + $i*a";
			}else {
				$command3 = $command3." ,$lambda^".abs($i-$ceil)."*a";
				$command1 = $command1." + $lambda^".abs($i-$ceil)."*a";
				$command2 = $command2." + $i*$lambda^".abs($i-$ceil)."*a";
			}			
		}	
		print("================\n");
		print($sim." One\n");
			
		
		my $result = "";
		if($sim != 0.0){
			print("Solve[".$command2." == $sim, {a}]\n");
			$result = $math->evaluate("Solve[".$command2." == $sim, {a}]");
		}else{
			print("Solve[$command1==1, {a}]\n");
			$result = $math->evaluate("Solve[$command1==1, {a}]");
		}
		my $numer = 0.0;
		if($result =~ /->\s+(\d\.\d+)\}/){
			$numer = $1;
		}else{
			print("error1");
			exit;
		}
		
		my $finalOut = "";
		for($i =$starter; $i <= $labelNo; $i++){
			if($i == $starter){
				$finalOut .= ($lambda**abs($i-$ceil))*$numer;
			}else{
				$finalOut .= "\t".(($lambda)**abs($i-$ceil))*$numer;
			}
		}
		print $result."\n";
		print $finalOut."\n";
		print OUT $finalOut."\n";
	}	
    else{
		for($i =$starter; $i <= $labelNo; $i++){
			if($i <= $floor){
				if ($i == $floor){
					$command3 = $command3." ,a ";
					$command1 = $command1." + a";					
					$command2 = $command2." + $i*a";
				}else {
					$command3 = $command3." ,$lambda^".abs($i-$floor)."*a";
					$command1 = $command1." + $lambda^".abs($i-$floor)."*a";
					$command2 = $command2." + $i*$lambda^".abs($i-$floor)."*a";
				}
			} else {
				if ($i == $ceil){
					$command3 = $command3." ,b";
					$command1 = $command1." + b";
					$command2 = $command2." + $i*b";
				}else {
					$command3 = $command3." ,$lambda^".abs($i-$ceil)."*b";
					$command1 = $command1." + $lambda^".abs($i-$ceil)."*b";
					$command2 = $command2." + $i*$lambda^".abs($i-$ceil)."*b";
				}
			}
		}
		
		$command3 .= "}"; 
		print("================\n");
		print($sim." Two\n");
		#print($command1."\n");
		#print($command2."\n");
		print("Solve[$command1==1&&$command2 == $sim, {a, b}]\n");		
		my $result = $math->evaluate("Solve[$command1==1&&$command2 == $sim, {a, b}]");
		
		my $numer1 = 0.0;
		my $numer2 = 0.0;
		if($result =~ /->\s+(\d\.\d+), b\s->\s(\d.\d+)\}/){
			$numer1 = $1;
			$numer2 = $2;
		}else{
			print("error2: $result");
			exit;
		}
		
		my $finalOut = "";
		for($i =$starter; $i <= $labelNo; $i++){
			if($i == $starter){
				$finalOut = (($lambda)**abs($i-$floor))*$numer1;
			}elsif($i <= $floor){
				$finalOut .= "\t".(($lambda)**abs($i-$floor))*$numer1;
			}else{
				$finalOut .= "\t".(($lambda)**abs($i-$ceil))*$numer2;
			}		
		}
		print $result;
		print $finalOut."\n";
		print OUT $finalOut."\n";		
	}	  
	print ($cc."\n");
	$cc = $cc + 1;
}

print STDERR $cc."\n";
close IN;
close OUT;
=cut
		for(my $i =$ceil; $i >= $starter; $i--){
			if ($i == $ceil){
				$command1 = "$i*a";
			} else {
				$command1 = " $i*$lambda^".($ceil-$i)."*a +" + $command1;
			}
		}
		
		for(my $i =$ceil+1; $i <= $labelNo; $i++){
			$command1 = $command1 + " + $i*$lambda^".($i-$ceil)."*a";			
		}
=cut