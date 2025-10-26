#!/bin/bash

current_project="$(pwd)"

whitelist_removal=()
# List of locally compiled files
elf_list=($(find ${current_project} -type f -exec file {} + | egrep 'ELF' | tr -d ' '))

# Create a list of only locally compiled files
for _elf in "${elf_list[@]}";
do
	whitelist_removal+=( "$(echo ${_elf} | cut -d':' -f1)" )
done

if test ${#whitelist_removal[@]} -gt 0;
then
	# Create a seperator
	for i in {0..50};do printf "\033[34m=\033[0m";done;echo;
	# Display a whitelist
	printf "\033[96mDo you want to remove the following ${#whitelist_removal[@]} compiled program(s):\033[0m\n"
	printf "\033[35mProgram:\t\033[33m%s\033[0m\n" ${whitelist_removal[@]}
	printf "\033[31m"
	read -p "Are you sure? [yes|no] " _confirm
	printf "\033[0m\n"
	case $_confirm in
		y|Y|yes|Yes) rm -v ${whitelist_removal[@]};;
		*) printf "Exiting, execution no files were removed.\n";;
	esac
else
	printf "\033[35mNo compiled file(s) were found.\033[0m\n"
fi

