#!/bin/bash

makefile_tmpl="Makefile.tmpl"
makefile_target="$(basename -s .tmpl ${makefile_tmpl})"

error(){
	printf "\033[35mError:\t\033[31m${1}!\033[0m\n"
	exit 1
}

define_program(){
	if [ -e "${makefile_tmpl}" ];
	then
		cp -a -v ${makefile_tmpl} ${makefile_target}
		printf "Please enter the name of the target: "
		read progname
		if [ -n "${progname}" -a -f "${makefile_target}" ];
		then
			_progname="$(basename -s .cu ${progname})"
			sed -i "s|{% PROGNAME %}|${_progname}|g" ${makefile_target}
		else
			error "Missing program name"
		fi
	else
		error "Unable to find template ${makefile_target}"
	fi
}

for argv in $@
do
	case $argv in
		-c|--create)
		define_program
		;;
	esac
done
