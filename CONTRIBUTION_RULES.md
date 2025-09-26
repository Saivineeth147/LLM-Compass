# Contribution Rules – Awesome LLM Resources

Thank you for contributing to **Awesome LLM Resources**! 🎉  
To keep the repository organized and maintain high-quality content, please follow these contribution rules.

---

## 1️⃣ Branch Naming Convention

All branches **must** follow the pattern:

`<type>/<short-description>` 


Where:  

- **`<type>`** indicates the purpose of the branch:  
  - `feature` → Adding new resources, tools, or sections  
  - `fix` → Bug fixes, broken links, or corrections  
  - `chore` → Maintenance tasks (CI/CD, configs, cleanup)  

- **`<short-description>`** → A brief, lowercase, hyphenated description of the branch’s purpose  

**Examples:**  
- `feature/add-auto-gpt-resource`  
- `fix/update-paper-link`  
- `chore/setup-ci-actions`  

---

## 2️⃣ Contribution Process

1. **Fork the Repository**  
   - Click “Fork” on the top right of the repo.  

2. **Create a Branch**  
   - Use the branch naming convention:  
     ```bash
     git checkout -b feature/add-resource
     ```

3. **Make Your Changes**  
   - Add or update resources in the README.  
   - Keep resources alphabetically sorted where relevant.  
   - Add a short description for new resources (1–2 lines max).  

4. **Commit Changes**  
   - Write clear commit messages:  
     ```bash
     git commit -m "Add [resource name] – short description"
     ```

5. **Push and Create PR**  
   - Push your branch:  
     ```bash
     git push origin <branch-name>
     ```  
   - Open a Pull Request (PR) to `main`.  
   - PRs must include a short description of the change.  

---

## 3️⃣ Pull Request Guidelines

- All PRs **must be reviewed** before merging.  
- Ensure no direct commits are made to `main`.  
- Keep PRs focused: one feature/fix per branch.  
- Check for typos, broken links, and correct formatting.  

---

## 4️⃣ Code of Conduct

- Be respectful and professional.  
- Constructive feedback is encouraged.  
- Avoid spam, promotional links, or unrelated content.  

