.PHONY: clean clean_debug_data blur_noise wiener_olimp wiener_skimage rich_lucy

clean:
	rm -rf results/

clean_debug_data:
	rm -rf results/blurred
	rm -rf results/restored_rich_lucy
	rm -rf results/restored_wiener_skimage
	rm -rf results/restored_wiener_olimp
	rm -rf results/psf

blur_noise:
	python blur_noise.py

wiener_olimp:
	python wiener_olimp.py
	python w_stat.py

wiener_skimage:
	python wiener_skimage.py
	python w_stat.py

rich_lucy:
	python rich_lucy.py
	python rl_stat.py

full_restorage:
	python wiener_olimp.py
	python w_stat.py
	python wiener_skimage.py
	python ws_stat.py
	python rich_lucy.py
	python rl_stat.py


full:
	python blur_noise.py
	python wiener_olimp.py
	python w_stat.py
	python wiener_skimage.py
	python ws_stat.py
	python rich_lucy.py
	python rl_stat.py