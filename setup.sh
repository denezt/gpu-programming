#!/bin/bash
#
#

driver_version=470
option="${1}"

error(){
	printf "\033[35mError\t\033[31m${1}\033[0m\n"
}

install(){
	requirements=( "nvidia-utils-${driver_version}" "cuda-nvcc-10-1" "cuda-drivers-${driver_version}" "nvidia-cuda-toolkit" "cuda" "cuda-drivers" )
	for req in ${requirements[*]};
	do
		printf "\033[35mInstalling, ${req}\033[0m\n"
		sudo apt-get install "${req}" --fix-missing -y
	done
	# Start NVIDIA Driver
	printf "\033[36mStarting, NVIDIA Driver...\033[0m\n"
	case "$(sudo prime-select query)" in
		'nvidia') printf "\033[32mNvidia is already selected.\033[0m\n";;
		*) sudo prime-select nvidia;;
	esac
	[ -z "$(lsmod | egrep -o 'nvidia' | head -n 1)" ] && sudo modprobe nvidia || printf "\033[32mModule nvidia is already activate.\033[0m\n"
	# software-properties-gtk
}

uninstall(){
	printf "\033[31mCaution, this will remove the previous install Cuda drivers and compilers!!!\033[0m\n"
	read -p "Are you sure? [yes|no] " _confirm
	case $_confirm in
		y|Y|yes)
		prime-select intel
		apt-get remove gcc g++ nvidia-* cuda* nvidia-cuda-toolkit -y
		apt-get purge nvidia-* gcc* g++* -y
		apt-get purge nvidia-cuda-toolkit -y
		apt autoremove -y
		;;
		*) printf "Exiting, no further execution\n";;
	esac
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
		--action=*) _action=$(echo $arg | cut -d'=' -f2);;
		--version=*) _version=$(echo $arg | cut -d'=' -f2);;
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
