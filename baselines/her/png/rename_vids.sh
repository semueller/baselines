mv_new_name(){
	OLDIFS=$IFS
	IFS='/'
	read -ra x <<< "$0"
	IFS=$OLDIFS
	newname="./${x[1]}_${x[2]}.mp4"
	echo "mv $0 $newname"
	mv $0 $newname
}

export -f mv_new_name

find -name _rollout.mp4 -type f -exec bash -c 'mv_new_name "$0"' {} \;
