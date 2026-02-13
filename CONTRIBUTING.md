# Contributing

感谢你对本项目的关注！欢迎通过 Issue 和 Pull Request 参与贡献。

## 开发流程

1. Fork 本仓库
2. 创建特性分支：`git checkout -b feature/your-feature`
3. 提交更改：`git commit -m "feat: add your feature"`
4. 推送分支：`git push origin feature/your-feature`
5. 创建 Pull Request

## 构建与测试

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
./mini_image_pipe_tests
```

## 代码规范

- C++/CUDA 代码遵循项目现有风格
- 使用 `.editorconfig` 中定义的缩进和格式规则
- 新增算子请附带单元测试
- 确保所有现有测试通过

## 提交信息格式

推荐使用 [Conventional Commits](https://www.conventionalcommits.org/)：

- `feat:` 新功能
- `fix:` 修复 Bug
- `docs:` 文档更新
- `perf:` 性能优化
- `test:` 测试相关
