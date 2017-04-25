use strict;
use warnings;


while(<>){
	chomp;
	if($_ > 0.5){
		printf("true\t%.4f\n", $_);
	}else{
		printf("false\t%.4f\n", $_);
	}
}
