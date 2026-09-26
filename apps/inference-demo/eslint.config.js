// @ts-check
import { defineConfig } from 'eslint/config'
import js from '@eslint/js'
import globals from 'globals'
import tseslint from 'typescript-eslint'

export default defineConfig(
  { ignores: ['dist/**', 'node_modules/**'] },
  js.configs.recommended,
  ...tseslint.configs.strictTypeChecked,
  {
    languageOptions: {
      ecmaVersion: 2022,
      globals: globals.browser,
      parserOptions: {
        projectService: {
          allowDefaultProject: ['*.config.ts', '*.config.js'],
        },
        tsconfigRootDir: import.meta.dirname,
      },
    },
  },
  {
    // Node-context config files (not part of tsconfig.json's "src" include).
    files: ['*.config.ts', '*.config.js'],
    languageOptions: {
      globals: globals.node,
    },
  },
  {
    // Test files run under Vitest's node environment.
    files: ['src/**/*.test.ts'],
    languageOptions: {
      globals: globals.node,
    },
  },
)
