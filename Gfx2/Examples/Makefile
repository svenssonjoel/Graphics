
DIRS = feigenbaum \
       image_test \
       info3ds \
       show_feig \
       space \
       space2 \
       simple2d \
       view3ds \
       viewmd2 \
       ground_mesh \
       dungeon

.PHONY : make_rec

make_rec :
	for d in $(DIRS); do (cd $$d; $(MAKE) all ); done


.PHONY : clean

clean : 
	for d in $(DIRS); do (cd $$d; $(MAKE) clean ); done