# Changelog

All notable changes to this project will be documented in this file. See [standard-version](https://github.com/conventional-changelog/standard-version) for commit guidelines.

### [0.2.1](https://github.com/biancarosa/cat-activities-monitor/compare/v0.2.0...v0.2.1) (2025-06-21)


### Features

* add cat identification debug script and fix feedback processing ([e619ed3](https://github.com/biancarosa/cat-activities-monitor/commit/e619ed3e8a8a5f18e4d35f160599301c6d3e922c))
* add cat identification training ([89ae78f](https://github.com/biancarosa/cat-activities-monitor/commit/89ae78f3e15d01a10a65ec8ec1c49da524d8bc3b))
* add class correction capability to YOLO feedback system ([67c2d38](https://github.com/biancarosa/cat-activities-monitor/commit/67c2d388733e4e00f7fb8f19e1dd1bb114a936df))
* **api:** migrate database service to SQLAlchemy ORM ([cba0394](https://github.com/biancarosa/cat-activities-monitor/commit/cba0394ce1014ea26a53c2f59e72fce489af8513))
* display unidentified cats in UI ([5fa916c](https://github.com/biancarosa/cat-activities-monitor/commit/5fa916c5428430fd36b09844e648203c38e3ba93))
* **ml:** feature extraction logs ([43d2a3a](https://github.com/biancarosa/cat-activities-monitor/commit/43d2a3aec4405fe38171a7a0d1d1ae7c4ffa2550))
* **ml:** implement cat recognition pipeline with feature extraction ([a3b902e](https://github.com/biancarosa/cat-activities-monitor/commit/a3b902e57cbb5eb4c2978b45e5b65f913b4c5b49))


### Bug Fixes

* Address linting issues in backend and frontend ([1d43727](https://github.com/biancarosa/cat-activities-monitor/commit/1d43727311070b91c4208b2efc6acf04592226a8))
* **api:** resolve ruff linting errors ([6ee5a18](https://github.com/biancarosa/cat-activities-monitor/commit/6ee5a18fef7186b403683dc6a396dceffac14a1a))
* handle list format in debug script database check ([aba2e25](https://github.com/biancarosa/cat-activities-monitor/commit/aba2e2504f36723ddef1244c9119177c90fe6da5))
* **ml:** add COCO_CLASSES to utils module for YOLO detection ([6997e3a](https://github.com/biancarosa/cat-activities-monitor/commit/6997e3ae92ee97850f0b671e3a9cacce31710240))
* persist cat_name field in detection database storage ([04ec667](https://github.com/biancarosa/cat-activities-monitor/commit/04ec6677797be9405f706cea7cf8d3a66a43b9c8))
* remove unused variable in settings page ([e0d6de6](https://github.com/biancarosa/cat-activities-monitor/commit/e0d6de612be6cce949bb3853e071e178ee97e9c4))
* resolve linting issues ([26a0a3f](https://github.com/biancarosa/cat-activities-monitor/commit/26a0a3fd97521fc5d551f53ec0ff10dbb719c94b))


### Performance Improvements

* remove features field from API responses ([d3b867d](https://github.com/biancarosa/cat-activities-monitor/commit/d3b867d8d0930cb2cfee7490c15440616e22140f))


### Code Refactoring

* **api:** remove dup variable & fix imports ([fee929a](https://github.com/biancarosa/cat-activities-monitor/commit/fee929adf1df8e3fd5648832e5259d5458ae0520))


### Documentation

* consolidate debug documentation and update project structure ([12cfa48](https://github.com/biancarosa/cat-activities-monitor/commit/12cfa48a048bf1386c2519aeee9eca3ee07f4090))
* update claude MD with docs folder ([642eaf3](https://github.com/biancarosa/cat-activities-monitor/commit/642eaf3bf3f2ed6ec120e9ad24d4963de9d9f360))
* update CLAUDE.md with session learnings and corrections ([b16c47b](https://github.com/biancarosa/cat-activities-monitor/commit/b16c47b395d47359649f5bd9f78e1b8f12f892fc))

## [0.2.0](https://github.com/biancarosa/cat-activities-monitor/compare/v0.1.2...v0.2.0) (2025-06-15)


### Features

* **backend:** alembic migrations ([d7b138e](https://github.com/biancarosa/cat-activities-monitor/commit/d7b138e47747e9ff515492bf1205f2eda1c17ac4))
* better feedback modal ([ea10282](https://github.com/biancarosa/cat-activities-monitor/commit/ea10282d593fab8ace51790f4ef8832229092bde))
* implement pagination for image gallery ([3478a55](https://github.com/biancarosa/cat-activities-monitor/commit/3478a556856ff4b8a379e8521d7f8289a627cb30))
* proper feedback linking to cat profiles ([a13184a](https://github.com/biancarosa/cat-activities-monitor/commit/a13184ad19236dcce219ad980e3cb596d14eadd7))
* use cat profile for bounding box ([2179a0e](https://github.com/biancarosa/cat-activities-monitor/commit/2179a0eeb20a519df17451fc79c9fdb9fa308b55))


### Bug Fixes

* **backend:** cat profile retrieval ([e523138](https://github.com/biancarosa/cat-activities-monitor/commit/e523138122f8924dff7b144ae5d5073f26e11592))
* **frontend:** build errors ([ecac781](https://github.com/biancarosa/cat-activities-monitor/commit/ecac78193b05a2aee2556d81b7d7e57fd0b30d73))
* **frontend:** use proper url to fetch static images ([16b0efd](https://github.com/biancarosa/cat-activities-monitor/commit/16b0efd08d251df123b7d30328158f580a9b57b6))
* include root package.json in version updates ([1c9d793](https://github.com/biancarosa/cat-activities-monitor/commit/1c9d79320048cb6a84eb7eb69376fff8dc9ad417))


### Chores

* gitignore api venv ([4c5666e](https://github.com/biancarosa/cat-activities-monitor/commit/4c5666e2f5311db86a54bb61fbd7e0e4a516dc7b))
* remove unused field ([e6ed3b9](https://github.com/biancarosa/cat-activities-monitor/commit/e6ed3b9515287d9395c472aa52a7a3bfde458b05))


### Code Refactoring

* **api:** delete a bunch of shitty activity detection code ([9f7fed8](https://github.com/biancarosa/cat-activities-monitor/commit/9f7fed814e06c37761cb3793d8cc27b3992f2816))
* **frontend:** comment training mgmt ([132b797](https://github.com/biancarosa/cat-activities-monitor/commit/132b797c99276ae62f5e3d36907e65abd1ccf360))
* rename detection_path to detection_imgs_path ([5e28caa](https://github.com/biancarosa/cat-activities-monitor/commit/5e28caa0b5aed5cfbcc9702ab245cc9eb99c9012))

### [0.1.2](https://github.com/biancarosa/cat-activities-monitor/compare/v0.1.1...v0.1.2) (2025-06-14)


### Features

* **ci:** add GitHub workflow for frontend linting ([21c96c7](https://github.com/biancarosa/cat-activities-monitor/commit/21c96c7628a4b1dacb12f4f8b183dfcc535daa48))
* **config:** add release as valid commit scope ([3dc774c](https://github.com/biancarosa/cat-activities-monitor/commit/3dc774c4f256fd488d6eb68f3506d299a241ac59))


### Bug Fixes

* **api:** resolve all ruff linting errors ([6adec0f](https://github.com/biancarosa/cat-activities-monitor/commit/6adec0f9a214d7ec7f0c68a53ebbd8eabb999989))
* **ci:** fix messy package-locks ([0c74d29](https://github.com/biancarosa/cat-activities-monitor/commit/0c74d292b04a600efffe9f7b386a5097a280eff2))
* **frontend:** frontend build fix ([e87859f](https://github.com/biancarosa/cat-activities-monitor/commit/e87859f2c93482672cde581829a08105930d2b11))
* **frontend:** node version for fe lint ([e778d2a](https://github.com/biancarosa/cat-activities-monitor/commit/e778d2a8b71f97026bee6c3f5c438768f2bbaa71))
* version on package.json and pyproject.toml ([61bde14](https://github.com/biancarosa/cat-activities-monitor/commit/61bde1442c28f5ad394a6e0acacfb3b53676f975))


### Continuous Integration

* **api:** add workflow to lint be ([299b418](https://github.com/biancarosa/cat-activities-monitor/commit/299b418fd4c1188b5adae0611c2fdc27f257a1df))


### Chores

* **api:** run ruff autofix ([c61104c](https://github.com/biancarosa/cat-activities-monitor/commit/c61104c73ad4a2cd00e06799f56d964b6a097ebb))
* **release:** null ([5a7e564](https://github.com/biancarosa/cat-activities-monitor/commit/5a7e5645c5ba35bddbcd8871cbf575ea951c96d9))

### [0.1.1](https://github.com/biancarosa/cat-activities-monitor/compare/v0.1.0...v0.1.1) (2025-06-14)


### Bug Fixes

* **frontend:** only use API url from settings ([38c069a](https://github.com/biancarosa/cat-activities-monitor/commit/38c069a002ae1dacafcc4fe5a09d46ebdfe1f3bb))


### Chores

* update configuration files ([fe7c9f7](https://github.com/biancarosa/cat-activities-monitor/commit/fe7c9f7aecf1aeba6ee9d8033967ca267a411121))


### Documentation

* update roadmap for individual cat recognition ([06c40c8](https://github.com/biancarosa/cat-activities-monitor/commit/06c40c8b105fd633a2a8108fdf8e7807cc083267))

## 0.1.0 (2025-06-13)


### Features

* initialize cat activities monitor project ([9649b4c](https://github.com/biancarosa/cat-activities-monitor/commit/9649b4cfc3eaf1053a1e986d9fc1bff33ce31571))


### Chores

* add conventional-commits ([b700559](https://github.com/biancarosa/cat-activities-monitor/commit/b700559a690551df75da51cd843771073346fe7f))
