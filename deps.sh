#!/bin/bash
#

option="${1}"

error(){
	printf "\033[35mError\t\033[31m${1}\033[0m\n"
}

install(){
	apt-get install nvidia-utils-455 --fix-missing -y
	apt install cuda-nvcc-10-1 --fix-missing -y
	apt-get install cuda-drivers-455 --fix-missing -y
	apt install nvidia-cuda-toolkit --fix-missing -y
	apt-get install cuda --fix-missing -y
	apt-get install cuda-drivers --fix-missing -y
	# Start NVIDIA Driver
	prime-select nvidia
	modprobe nvidia
	# software-properties-gtk
	}

uninstall(){
	prime-select intel
	apt-get remove gcc -y
	apt-get remove g++ -y
	apt-get remove nvidia-* -y
	apt-get purge nvidia-* gcc* g++* -y
	apt-get remove cuda* -y
	apt-get remove nvidia-cuda-toolkit -y
	apt-get purge nvidia-cuda-toolkit -y
	apt autoremove -y
}

change_compiler(){
	version=${1}
	dir="/usr/bin"
	prog=( 'gcc' 'g++' )
	case $version in
		8|9|10) 
		if [ ! -z "${version}" ];
		then
			for _prog in ${prog[*]}
			do
				if test -f "${dir}/${_prog}-${version}";
				then
					printf "Linking: ${dir}/${_prog}-${version} => ${dir}/${_prog}\n"
					ln -f -s ${dir}/${_prog}-${version} ${dir}/${_prog}
				else
					error "Program ${dir}/${_prog}-${version} was not found or does not exist!"
				fi
			done
		fi
		;;
		*) error "Missing or invalid target version for compiler." && [ ! -z "${version}" ] && error "Version $version is not available";;
	esac
	return 0
}


configure(){
	if [ -f "blacklist-nouveau.conf" ];
	then
		cp -a -v "blacklist-nouveau.conf" "/etc/modprobe.d/"
		update-initramfs -u
	fi
}


help_menu(){
	printf "\033[36mNvidia Quick Install Wrapper\033[0m\n"
	printf "\033[1;2;33mPARAMETERS\033[0m\n"
	printf "\033[1;2;35mSet the action\t\t\033[32maction\033[0m\n"
	printf "\033[1;2;35mSet the target version\t\033[32mversion\033[0m\n"
	printf "\n\033[1;2;33mACTIONS\033[0m\n"
	printf "\033[1;2;35mInstall:\t\t\033[1;3;34mi, install\033[0m\n"
	printf "\033[1;2;35mUninstall:\t\t\033[1;3;34mu, uninstall\033[0m\n"
	printf "\033[1;2;35mChange Compilers:\t\033[1;3;34mc, cc, change-compiler\033[0m\n"
	printf "\n\033[1;2;33m\nUSAGE:\033[0m\n"
	printf "$0 --action=install # Will Install the nvidia version\n"
	printf "$0 --action=unstall # Will Install the nvidia version\n"
	printf "$0 --action=change-compiler --version=8 # Will Change the nvidia version\n"
	exit 0
}

for arg in $@
do
	case $arg in
		--action=*) _action=$(echo $arg| cut -d'=' -f2);;
		--version=*) _version=$(echo $arg| cut -d'=' -f2);;
		-h|-help|--help) help_menu;;
	esac
done

case $_action in
	i|install)
	install
	configure
	;;
	u|uninstall) uninstall;;
	c|cc|change-compiler) change_compiler "${_version}";;
esac

###############		END OF SCRIPT		###############
